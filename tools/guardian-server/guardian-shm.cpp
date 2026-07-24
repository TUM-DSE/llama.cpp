#include "guardian-shm.h"

#include "guardian/shm.h" // canonical ABI header from the guardian repo (native/)

#include "server-common.h"
#include "server-schema.h"
#include "server-task.h"
#include "log.h"

#include <sys/io.h>
#include <cerrno>
#include <cstring>
#include <ctime>

// event values on BENCHMARK_PORT (0xf4), matching the legacy guardian-server:
//   201 = request picked up from shm, 202 = result written back to shm
#define GUARDIAN_EVENT_PICKUP 201
#define GUARDIAN_EVENT_DONE   202

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

void guardian_shm_frontend::scan_loop() {
    SharedMemory * shm = (SharedMemory *) comm->requestQueue;

    while (running.load()) {
        // timed wait instead of the legacy plain sem_wait so shutdown is clean
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 1;
        if (sem_timedwait(&shm->active_reqs, &ts) != 0) {
            continue; // ETIMEDOUT / EINTR — re-check running
        }
        // one active_reqs post corresponds to one pending serverNotifier
        for (int i = 0; i < MAX_REQUESTS; i++) {
            if (sem_trywait(&shm->requests[i].serverNotifier) == 0) {
                std::thread(&guardian_shm_frontend::handle_request, this, i).detach();
                break;
            }
        }
    }
}

// write the response into the slot's text field (bounded) and wake the client
static void guardian_finish(SharedMemory * shm, int slot_idx, const std::string & body,
                            bool io_ok, int outb_id) {
    Request & req = shm->requests[slot_idx];

    size_t n = body.size();
    if (n >= (size_t) MAX_TEXT) {
        SRV_WRN("guardian: response truncated (%zu > %d)\n", n, MAX_TEXT - 1);
        n = MAX_TEXT - 1;
    }
    memcpy(req.text, body.data(), n);
    req.text[n] = '\0';

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
        guardian_finish(shm, slot_idx,
                        safe_json_to_str({{"error", "invalid guardian request JSON"}}),
                        io_ok, outb_id);
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

        auto inputs = tokenize_input_prompts(vocab, /*mctx=*/nullptr, data.at("prompt"),
                                             /*add_special=*/true, /*parse_special=*/true);
        task.tokens = std::move(inputs[0]);

        task.params = server_schema::eval_llama_cmpl_schema(
                vocab, *params_base, meta->slot_n_ctx, meta->logit_bias_eog, data);
        task.params.stream   = false;
        task.params.n_cmpl   = 1;
        task.params.res_type = TASK_RESPONSE_TYPE_NONE; // llama.cpp native JSON

        rd.post_task(std::move(task), /*front=*/prio <= 1);

        auto all = rd.wait_for_all([this]() { return !running.load(); });

        json out;
        if (all.error != nullptr) {
            out = all.error->to_json();
        } else if (all.is_terminated || all.results.empty()) {
            out = {{"error", "guardian-server terminated before completion"}};
        } else {
            out = all.results[0]->to_json();
            // also surface metrics in the fixed slot fields (additive; the
            // legacy server left them zeroed and clients parse the JSON body)
            if (auto * fin = dynamic_cast<server_task_result_cmpl_final *>(all.results[0].get())) {
                req.tokens_per_second = (float) fin->timings.predicted_per_second;
                req.decode_time_ms    = (float) fin->timings.predicted_ms;
            }
        }
        guardian_finish(shm, slot_idx,
                        out.dump(-1, ' ', false, json::error_handler_t::replace),
                        io_ok, outb_id);
    } catch (const std::exception & e) {
        SRV_ERR("guardian: request on slot %d failed: %s\n", slot_idx, e.what());
        guardian_finish(shm, slot_idx,
                        safe_json_to_str({{"error", e.what()}}),
                        io_ok, outb_id);
    }
}
