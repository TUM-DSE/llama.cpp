#include "guardian-shm.h"

#include "guardian/shm.h" // canonical ABI header from the guardian repo (native/)

#include "server-common.h"
#include "server-schema.h"
#include "server-task.h"
#include "log.h"

#include <sys/io.h>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

// event values on BENCHMARK_PORT (0xf4), matching the legacy guardian-server:
//   201 = request picked up from shm, 202 = result written back to shm
#define GUARDIAN_EVENT_PICKUP 201
#define GUARDIAN_EVENT_DONE   202

static long guardian_env_seconds(const char * name, long default_value) {
    const char * value = getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    char * end = nullptr;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed <= 0) {
        return default_value;
    }
    return parsed;
}

static long guardian_request_timeout_seconds() {
    return guardian_env_seconds("GUARDIAN_REQUEST_TIMEOUT_SECONDS", 900);
}

// How long a slot may stay in flight before the watchdog dumps the in-flight
// table. Must stay below the request timeout so a stall is reported while the
// request is still pending rather than after it has already been answered with
// an error.
static long guardian_stall_warn_seconds() {
    return guardian_env_seconds("GUARDIAN_STALL_WARN_SECONDS", 60);
}

guardian_shm_frontend::guardian_shm_frontend() : comm(new Communication{}) {}

guardian_shm_frontend::~guardian_shm_frontend() {
    stop();
}

bool guardian_shm_frontend::init() {
    if (init_shm(true, comm.get()) != 0) {
        SRV_ERR("%s", "guardian: failed to initialize shared memory\n");
        return false;
    }
    SRV_INF("guardian: shm region ready (%zu request slots, %d byte text field)\n",
            (size_t) MAX_REQUESTS, MAX_TEXT);
    return true;
}

void guardian_shm_frontend::start(server_context & ctx, const common_params & params) {
    ctx_server   = &ctx;
    params_base  = &params;
    meta         = std::make_unique<server_context_meta>(ctx.get_meta());

    // called once from the main thread before start_loop(), per server-context.h
    llama_context * lctx = ctx.get_llama_context();
    GGML_ASSERT(lctx != nullptr && "guardian: model must be loaded before start()");
    vocab = llama_model_get_vocab(llama_get_model(lctx));

    running.store(true);
    scanner = std::thread(&guardian_shm_frontend::scan_loop, this);
    SRV_INF("%s", "guardian: shm front-end started\n");
}

void guardian_shm_frontend::stop() {
    if (!running.exchange(false)) {
        return;
    }
    if (scanner.joinable()) {
        scanner.join();
    }
}

void guardian_shm_frontend::inflight_begin(int slot_idx, int req_id, int outb_id) {
    std::lock_guard<std::mutex> lock(inflight_mutex);
    inflight[slot_idx] = inflight_entry{req_id, outb_id, -1,
                                        std::chrono::steady_clock::now()};
}

void guardian_shm_frontend::inflight_set_task(int slot_idx, int task_id) {
    std::lock_guard<std::mutex> lock(inflight_mutex);
    auto it = inflight.find(slot_idx);
    if (it != inflight.end()) {
        it->second.task_id = task_id;
    }
}

void guardian_shm_frontend::report_stalls() {
    const auto now = std::chrono::steady_clock::now();
    const auto threshold = std::chrono::seconds(guardian_stall_warn_seconds());
    SharedMemory * shm = (SharedMemory *) comm->requestQueue;

    // Undispatched notifiers are tracked on EVERY tick, independently of the
    // in-flight map. A request the scanner never picked up has no in-flight
    // entry at all, so gating this sweep on a stale in-flight entry would miss
    // precisely the case the watchdog exists to catch.
    //
    // A single observation proves nothing — a client can post between the
    // dispatch sweep and this one — so a slot must stay posted across the
    // whole threshold before it is reported.
    std::vector<int> stuck_notifiers;
    for (int i = 0; i < MAX_REQUESTS; i++) {
        int value = 0;
        if (sem_getvalue(&shm->requests[i].serverNotifier, &value) == 0 && value > 0) {
            const auto inserted = notifier_pending_since.emplace(i, now);
            if (now - inserted.first->second >= threshold) {
                stuck_notifiers.push_back(i);
            }
        } else {
            notifier_pending_since.erase(i);
        }
    }

    size_t n_stalled = 0;
    {
        std::lock_guard<std::mutex> lock(inflight_mutex);
        for (const auto & [slot_idx, entry] : inflight) {
            if (now - entry.started >= threshold) {
                n_stalled++;
            }
        }
    }

    if (n_stalled == 0 && stuck_notifiers.empty()) {
        return;
    }
    // rate-limit the report only; the tracking above must keep running
    if (now - last_stall_report < threshold) {
        return;
    }
    last_stall_report = now;

    if (n_stalled > 0) {
        std::lock_guard<std::mutex> lock(inflight_mutex);
        SRV_WRN("guardian: %zu in-flight request(s) stalled >%llds (%zu in flight)\n",
                n_stalled, (long long) threshold.count(), inflight.size());
        for (const auto & [slot_idx, entry] : inflight) {
            if (now - entry.started < threshold) {
                continue;
            }
            const auto age = std::chrono::duration_cast<std::chrono::seconds>(
                    now - entry.started).count();
            SRV_WRN("guardian:   stalled shm_slot=%d req_id=%d outb_id=%d "
                    "llama_task=%d age=%llds\n",
                    slot_idx, entry.req_id, entry.outb_id, entry.task_id,
                    (long long) age);
        }
    }

    int credits = 0;
    sem_getvalue(&shm->active_reqs, &credits);
    if (!stuck_notifiers.empty()) {
        std::string slots;
        for (int slot_idx : stuck_notifiers) {
            const auto age = std::chrono::duration_cast<std::chrono::seconds>(
                    now - notifier_pending_since[slot_idx]).count();
            slots += " " + std::to_string(slot_idx) + "(" +
                     std::to_string((long long) age) + "s)";
        }
        SRV_WRN("guardian: %zu serverNotifier(s) posted but never dispatched, "
                "active_reqs=%d, slots:%s\n",
                stuck_notifiers.size(), credits, slots.c_str());
    } else {
        SRV_WRN("guardian: no undispatched serverNotifier (active_reqs=%d)\n",
                credits);
    }
}

void guardian_shm_frontend::scan_loop() {
    SharedMemory * shm = (SharedMemory *) comm->requestQueue;

    while (running.load()) {
        // `active_reqs` is a latency hint, not the dispatch authority: it wakes
        // us as soon as a client posts. The sweep below runs on EVERY wake,
        // including the ETIMEDOUT one, and dispatches every pending notifier —
        // so a credit that is lost, dropped or duplicated costs at most one
        // second of latency instead of stranding a request forever.
        //
        // The legacy loop consumed exactly one credit, dispatched the first
        // pending notifier and broke, assuming a perfect 1:1 pairing; a wake
        // that found no notifier silently discarded the credit with no
        // recovery path. See tests/golden/deviations.md.
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 1;
        sem_timedwait(&shm->active_reqs, &ts); // ETIMEDOUT / EINTR are expected

        // drain any surplus credits so the semaphore cannot grow without bound
        // when one wake dispatches several requests
        while (sem_trywait(&shm->active_reqs) == 0) {
        }

        for (int i = 0; i < MAX_REQUESTS; i++) {
            if (sem_trywait(&shm->requests[i].serverNotifier) != 0) {
                continue;
            }
            Request & req = shm->requests[i];
            SRV_INF("guardian: pickup shm_slot=%d req_id=%d outb_id=%d bytes=%zu\n",
                    i, req.id, req.outb_id, strnlen(req.text, MAX_TEXT));
            inflight_begin(i, req.id, req.outb_id);
            std::thread(&guardian_shm_frontend::handle_request, this, i).detach();
        }

        report_stalls();
    }
}

// write the response into the slot's text field (bounded), retire the slot from
// the in-flight table and wake the client
void guardian_shm_frontend::finish_request(int slot_idx, const std::string & body,
                                           bool io_ok, int outb_id,
                                           const char * outcome) {
    SharedMemory * shm = (SharedMemory *) comm->requestQueue;
    Request & req = shm->requests[slot_idx];

    size_t n = body.size();
    if (n >= (size_t) MAX_TEXT) {
        SRV_WRN("guardian: response truncated (%zu > %d)\n", n, MAX_TEXT - 1);
        n = MAX_TEXT - 1;
    }
    memcpy(req.text, body.data(), n);
    req.text[n] = '\0';

    long long elapsed_ms = -1;
    int req_id = -1;
    int task_id = -1;
    {
        std::lock_guard<std::mutex> lock(inflight_mutex);
        auto it = inflight.find(slot_idx);
        if (it != inflight.end()) {
            req_id  = it->second.req_id;
            task_id = it->second.task_id;
            elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - it->second.started).count();
            inflight.erase(it);
        }
    }
    SRV_INF("guardian: done shm_slot=%d req_id=%d outb_id=%d llama_task=%d "
            "elapsed_ms=%lld bytes=%zu outcome=%s\n",
            slot_idx, req_id, outb_id, task_id, elapsed_ms, n, outcome);

    if (io_ok && outb_id != -1) {
        my_outl(outb_id, GUARDIAN_EVENT_DONE);
    }
    sem_post(&req.clientNotifier);
}

void guardian_shm_frontend::handle_request(int slot_idx) {
    SharedMemory * shm = (SharedMemory *) comm->requestQueue;
    Request & req = shm->requests[slot_idx];

    // I/O-port permission is per-thread; degrade to no instrumentation if unavailable
    bool io_ok = ioperm(BENCHMARK_PORT, 4, 1) == 0;

    const int outb_id = req.outb_id;

    json data = json::parse(req.text, nullptr, /*allow_exceptions=*/false);
    if (data.is_discarded() || !data.contains("prompt")) {
        SRV_ERR("guardian: slot %d carries invalid request JSON\n", slot_idx);
        finish_request(slot_idx,
                       safe_json_to_str({{"error", "invalid guardian request JSON"}}),
                       io_ok, outb_id, "invalid-json");
        return;
    }

    // legacy prio: 0..9, lower = higher priority; the queue offers front/back
    const int prio = json_value(data, "prio", 9);

    if (io_ok && outb_id != -1) {
        my_outl(outb_id, GUARDIAN_EVENT_PICKUP);
    }

    try {
        auto rd = ctx_server->get_response_reader();

        server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
        task.id = rd.get_new_id();
        // the shm-slot → llama-task link: a pickup line with no matching
        // "new prompt" for this task id means llama.cpp holds the request
        // (deferred queue), not that the scanner missed it
        inflight_set_task(slot_idx, task.id);
        SRV_INF("guardian: posting shm_slot=%d req_id=%d -> llama_task=%d\n",
                slot_idx, req.id, task.id);

        auto inputs = tokenize_input_prompts(vocab, /*mctx=*/nullptr, data.at("prompt"),
                                             /*add_special=*/true, /*parse_special=*/true);
        task.tokens = std::move(inputs[0]);

        task.params = server_schema::eval_llama_cmpl_schema(
                vocab, *params_base, meta->slot_n_ctx, meta->logit_bias_eog, data);
        task.params.stream   = false;
        task.params.n_cmpl   = 1;
        task.params.res_type = TASK_RESPONSE_TYPE_NONE; // llama.cpp native JSON

        rd.post_task(std::move(task), /*front=*/prio <= 1);

        const auto deadline = std::chrono::steady_clock::now() +
                std::chrono::seconds(guardian_request_timeout_seconds());
        auto all = rd.wait_for_all([this, deadline]() {
            return !running.load() ||
                    std::chrono::steady_clock::now() >= deadline;
        });

        json out;
        const char * outcome = "ok";
        if (all.error != nullptr) {
            out = all.error->to_json();
            outcome = "llama-error";
        } else if (all.is_terminated && running.load()) {
            out = {{"error",
                    "guardian-server timed out waiting for llama.cpp completion"}};
            outcome = "llama-timeout";
        } else if (all.is_terminated || all.results.empty()) {
            out = {{"error", "guardian-server terminated before completion"}};
            outcome = "terminated";
        } else {
            out = all.results[0]->to_json();
            // also surface metrics in the fixed slot fields (additive; the
            // legacy server left them zeroed and clients parse the JSON body)
            if (auto * fin = dynamic_cast<server_task_result_cmpl_final *>(all.results[0].get())) {
                req.tokens_per_second = (float) fin->timings.predicted_per_second;
                req.decode_time_ms    = (float) fin->timings.predicted_ms;
            }
        }
        finish_request(slot_idx,
                       out.dump(-1, ' ', false, json::error_handler_t::replace),
                       io_ok, outb_id, outcome);
    } catch (const std::exception & e) {
        SRV_ERR("guardian: request on slot %d failed: %s\n", slot_idx, e.what());
        finish_request(slot_idx, safe_json_to_str({{"error", e.what()}}),
                       io_ok, outb_id, "exception");
    }
}
