// A basic application simulating a server with multiple clients.
// The clients submit requests to the server and they are processed in parallel.
// Clients are assigned priorities (0=highest) and the server uses priority scheduling
// with preemption: a higher-priority request can evict a lower-priority one from its slot.
//
// Requests arrive over time (not all at once) to demonstrate live preemption:
//   - First n_parallel requests arrive immediately at priority 2 (low) to fill all slots
//   - Subsequent requests arrive every --arrival-interval-ms ms with random priorities
//   - High-priority arrivals preempt running low-priority clients

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <cstdlib>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <ctime>
#include <algorithm>

// trim whitespace from the beginning and end of a string
static std::string trim(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();

    while (start < end && isspace(str[start])) {
        start += 1;
    }

    while (end > start && isspace(str[end - 1])) {
        end -= 1;
    }

    return str.substr(start, end - start);
}

static std::string k_system =
R"(Transcript of a never ending dialog, where the User interacts with an Assistant.
The Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User: Recommend a nice restaurant in the area.
Assistant: I recommend the restaurant "The Golden Duck". It is a 5 star restaurant with a great view of the city. The food is delicious and the service is excellent. The prices are reasonable and the portions are generous. The restaurant is located at 123 Main Street, New York, NY 10001. The phone number is (212) 555-1234. The hours are Monday through Friday from 11:00 am to 10:00 pm. The restaurant is closed on Saturdays and Sundays.
User: Who is Richard Feynman?
Assistant: Richard Feynman was an American physicist who is best known for his work in quantum mechanics and particle physics. He was awarded the Nobel Prize in Physics in 1965 for his contributions to the development of quantum electrodynamics. He was a popular lecturer and author, and he wrote several books, including "Surely You're Joking, Mr. Feynman!" and "What Do You Care What Other People Think?".
User:)";

static std::vector<std::string> k_prompts = {
    "What is the meaning of life?",
    "Tell me an interesting fact about llamas.",
    "What is the best way to cook a steak?",
    "Are you familiar with the Special Theory of Relativity and can you explain it to me?",
    "Recommend some interesting books to read.",
    "What is the best way to learn a new language?",
    "How to get a job at Google?",
    "If you could have any superpower, what would it be?",
    "I want to learn how to play the piano.",
};

// A request waiting to arrive or in the priority queue before being assigned to a slot.
struct pending_req {
    int32_t seq_id   = -1;
    int32_t priority = 0;   // lower = higher priority
    int64_t t_arrive = 0;   // ggml_time_us() when this request becomes visible to the scheduler
    int64_t t_queued = 0;   // ggml_time_us() when moved into the pending queue (= t_arrive)
    std::string input;
    std::string prompt;
};

struct client {
    ~client() {
        if (smpl) {
            common_sampler_free(smpl);
        }
    }

    int32_t id = 0;

    llama_seq_id seq_id = -1;

    llama_token sampled;

    int64_t t_start_prompt;
    int64_t t_start_gen;

    int32_t n_prompt  = 0;
    int32_t n_decoded = 0;
    int32_t i_batch   = -1;

    std::string input;
    std::string prompt;
    std::string response;

    struct common_sampler * smpl = nullptr;

    // priority scheduling
    int32_t priority        = 0;  // lower = higher priority
    int64_t t_queued        = 0;  // when request entered the pending queue
    int64_t t_first_sched   = 0;  // when first assigned to a slot (end of initial wait)
    int64_t t_preempt_total = 0;  // total time (us) spent preempted across all preemptions
    int64_t t_preempt_start = 0;  // start of current preemption period (0 = not preempted)
    int32_t n_preemptions   = 0;
    llama_seq_id park_seq   = -1; // KV seq_id used for parking while preempted
};

static void print_date_time() {
    std::time_t current_time = std::time(nullptr);
    std::tm* local_time = std::localtime(&current_time);
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", local_time);

    LOG_INF("\n");
    LOG_INF("\033[35mrun parameters as of %s\033[0m\n", buffer);
    LOG_INF("\n");
}

// Define a split string function to ...
static std::vector<std::string> split_string(const std::string& input, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream stream(input);
    std::string token;
    while (std::getline(stream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

int main(int argc, char ** argv) {
    // Seed is set later, after model load, using ggml_time_us() for variation

    // Extract --arrival-interval-ms before common_params_parse sees unknown args
    int64_t arrival_interval_us = 2000000LL; // 2 seconds default
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--arrival-interval-ms" && i + 1 < argc) {
            arrival_interval_us = std::stoll(argv[++i]) * 1000LL;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    int filtered_argc = (int)filtered_argv.size();

    common_params params;

    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_PARALLEL)) {
        return 1;
    }

    common_init();

    // number of simultaneous "clients" to simulate
    const int32_t n_clients = params.n_parallel;

    params.n_parallel += n_clients + 1;

    // requests to simulate
    const int32_t n_seq = params.n_sequences;

    // insert new requests as soon as the previous one is done
    const bool cont_batching = params.cont_batching;

    const bool dump_kv_cache = params.dump_kv_cache;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the target model
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // load the prompts from an external file if there are any
    if (params.prompt.empty()) {
        LOG_INF("\033[32mNo new questions so proceed with build-in defaults.\033[0m\n");
    } else {
        // Output each line of the input params.prompts vector and copy to k_prompts
        int index = 0;
        LOG_INF("\033[32mNow printing the external prompt file %s\033[0m\n\n", params.prompt_file.c_str());

        std::vector<std::string> prompts = split_string(params.prompt, '\n');
        for (const auto& prompt : prompts) {
            k_prompts.resize(index + 1);
            k_prompts[index] = prompt;
            index++;
            LOG_INF("%3d prompt: %s\n", index, prompt.c_str());
        }
    }

    LOG_INF("\n\n");

    const int n_ctx = llama_n_ctx(ctx);

    std::vector<client> clients(n_clients);
    for (size_t i = 0; i < clients.size(); ++i) {
        auto & c = clients[i];
        c.id   = i;
        c.smpl = common_sampler_init(model, params.sampling);
    }

    std::vector<llama_token> tokens_system;
    tokens_system = common_tokenize(ctx, k_system, true);
    const int32_t n_tokens_system = tokens_system.size();

    // the max batch size is as large as the context to handle cases where we get very long input prompt from multiple
    // users. regardless of the size, the main loop will chunk the batch into a maximum of params.n_batch tokens at a time
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    int32_t n_total_prompt = 0;
    int32_t n_total_gen    = 0;
    int32_t n_cache_miss   = 0;

    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx, n_clients);

    const auto t_main_start = ggml_time_us();

    LOG_INF("%s: Simulating parallel requests from clients:\n", __func__);
    LOG_INF("%s: n_parallel = %d, n_sequences = %d, cont_batching = %d, system tokens = %d\n",
            __func__, n_clients, n_seq, cont_batching, n_tokens_system);
    LOG_INF("%s: arrival_interval = %.2f s (use --arrival-interval-ms N to change)\n",
            __func__, arrival_interval_us / 1e6);
    LOG_INF("\n");

    {
        LOG_INF("%s: Evaluating the system prompt ...\n", __func__);

        for (int32_t i = 0; i < n_tokens_system; ++i) {
            common_batch_add(batch, tokens_system[i], i, { 0 }, false);
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return 1;
        }

        // assign the system KV cache to all active slot sequences
        for (int32_t i = 1; i <= n_clients; ++i) {
            llama_kv_self_seq_cp(ctx, 0, i, -1, -1);
        }

        LOG_INF("\n");
    }

    const int64_t t_arrivals_base = ggml_time_us();
    // Mix time with a Knuth multiplicative hash for a better-distributed seed.
    // t_arrivals_base varies by model-load duration (~ms), giving different seeds.
    unsigned int rng_seed = (unsigned int)(t_arrivals_base ^ (t_arrivals_base >> 17));
    rng_seed ^= rng_seed >> 16;
    rng_seed *= 0x45d9f3bU;
    rng_seed ^= rng_seed >> 16;
    srand(rng_seed);

    // Staggered requests cycle through priorities 0→1→2→0... starting at a
    // run-varying offset so preemption is always exercised but varies each run.
    const int prio_cycle_start = (int)(rng_seed % 3);

    std::vector<pending_req> future_reqs;
    future_reqs.reserve(n_seq);
    for (int32_t i = 0; i < n_seq; ++i) {
        pending_req req;
        req.seq_id = i;
        req.input  = k_prompts[rand() % k_prompts.size()];
        req.prompt = req.input + "\nAssistant:";
        if (i < n_clients) {
            // Fill all slots at t=0 with low-priority work
            req.priority = 2;
            req.t_arrive = t_arrivals_base;
        } else {
            // Cycle 0→1→2→0... so every run has at least one high-priority arrival
            const int stagger_idx = i - n_clients;
            req.priority = (prio_cycle_start + stagger_idx) % 3;
            req.t_arrive = t_arrivals_base + (int64_t)(stagger_idx + 1) * arrival_interval_us;
        }
        future_reqs.push_back(req);
    }
    // Keep future_reqs sorted by arrival time for efficient front-popping
    std::sort(future_reqs.begin(), future_reqs.end(),
        [](const pending_req & a, const pending_req & b) { return a.t_arrive < b.t_arrive; });

    // Pending queue: requests that have arrived and are waiting for a slot
    std::vector<pending_req> pending;

    // Preempted clients waiting to resume (hold full state + parked KV)
    std::vector<client> preempted;

    // Free-list for parking KV sequence IDs (n_clients+1 .. 2*n_clients)
    std::vector<llama_seq_id> park_ids_free;
    for (int i = n_clients + 1; i <= 2 * n_clients; ++i) {
        park_ids_free.push_back(i);
    }

    // Returns index of element with lowest .priority value, or -1 if empty
    auto best_idx = [](const auto & vec) -> int {
        if (vec.empty()) return -1;
        int best = 0;
        for (int i = 1; i < (int)vec.size(); ++i) {
            if (vec[i].priority < vec[best].priority) best = i;
        }
        return best;
    };

    // Returns index of active client with highest .priority number (= lowest scheduling prio), or -1
    auto worst_active_idx = [&]() -> int {
        int worst = -1, worst_prio = -1;
        for (int i = 0; i < (int)clients.size(); ++i) {
            if (clients[i].seq_id != -1 && clients[i].priority > worst_prio) {
                worst_prio = clients[i].priority;
                worst = i;
            }
        }
        return worst;
    };

    LOG_INF("Processing requests with priority scheduling and preemption...\n");
    LOG_INF("Priority legend: 0 = highest, 1 = medium, 2 = lowest\n\n");

    bool was_idle = false;

    while (true) {
        if (dump_kv_cache) {
            llama_kv_cache_view_update(ctx, &kvc_view);
            common_kv_cache_dump_view_seqs(kvc_view, 40);
        }

        // Flush requests whose arrival time has passed into the pending queue
        {
            const int64_t now = ggml_time_us();
            while (!future_reqs.empty() && future_reqs.front().t_arrive <= now) {
                pending_req & r = future_reqs.front();
                r.t_queued = r.t_arrive; // queue wait starts at arrival, not flush time
                LOG_INF("\033[36m[t=%.2f s] Request seq %4d arrived (priority %d)\033[0m\n",
                        (now - t_arrivals_base) / 1e6, r.seq_id, r.priority);
                pending.push_back(std::move(r));
                future_reqs.erase(future_reqs.begin());
            }
        }

        common_batch_clear(batch);

        // Determine if all slots are currently idle
        bool all_idle = true;
        for (const auto & c : clients) {
            if (c.seq_id != -1) { all_idle = false; break; }
        }

        if (all_idle && !was_idle) {
            // Transition to idle: clear active slot KV caches (but NOT parking slots)
            for (int i = 1; i <= n_clients; ++i) {
                llama_kv_self_seq_rm(ctx, i, -1, -1);
                llama_kv_self_seq_cp(ctx, 0, i, -1, -1);
            }
            was_idle = true;
        } else if (!all_idle) {
            was_idle = false;
        }

        if (cont_batching || all_idle) {

            for (auto & c : clients) {
                if (c.seq_id != -1) continue;

                const int pi = best_idx(preempted);
                const int qi = best_idx(pending);

                if (pi == -1 && qi == -1) break;

                const bool use_preempted = (pi != -1) &&
                    (qi == -1 || preempted[pi].priority <= pending[qi].priority);

                if (use_preempted) {
                    // --- Restore preempted client to this free slot ---
                    client & saved = preempted[pi];
                    const int64_t now = ggml_time_us();

                    saved.t_preempt_total += (now - saved.t_preempt_start);
                    saved.t_preempt_start  = 0;

                    // Move KV cache: parking seq → this slot's seq
                    const int slot_seq = c.id + 1;
                    const int32_t n_kv_resume = n_tokens_system + saved.n_prompt + saved.n_decoded;
                    {
                        const int64_t t_kv0 = ggml_time_us();
                        llama_kv_self_seq_rm(ctx, slot_seq, -1, -1);
                        llama_kv_self_seq_cp(ctx, saved.park_seq, slot_seq, -1, -1);
                        llama_kv_self_seq_rm(ctx, saved.park_seq, -1, -1);
                        LOG_INF("[t=%6.2f s] CTX_SWITCH type=resume    n_tokens=%5d elapsed_us=%d\n",
                                (now - t_arrivals_base) / 1e6, n_kv_resume,
                                (int)(ggml_time_us() - t_kv0));
                    }
                    park_ids_free.push_back(saved.park_seq);
                    saved.park_seq = -1;

                    LOG_INF("\033[32m[t=%6.2f s] Client %3d: RESUMED   seq %4d (priority %d) — "
                            "preempted %dx, total %.2f s\033[0m\n",
                            (now - t_arrivals_base) / 1e6,
                            c.id, saved.seq_id, saved.priority,
                            saved.n_preemptions, saved.t_preempt_total / 1e6);

                    // Restore state into this slot (preserve physical id and sampler)
                    struct common_sampler * slot_smpl = c.smpl;
                    const int32_t slot_id = c.id;
                    c      = saved;      // copies all fields (saved.smpl == nullptr)
                    c.id   = slot_id;
                    c.smpl = slot_smpl;  // reuse the slot's sampler
                    common_sampler_reset(c.smpl);

                    // Add the pending sampled token to the batch
                    c.i_batch = batch.n_tokens;
                    common_batch_add(batch, c.sampled,
                                     n_tokens_system + c.n_prompt + c.n_decoded,
                                     { c.id + 1 }, true);
                    c.n_decoded += 1;

                    preempted.erase(preempted.begin() + pi);

                } else {
                    // --- Assign new pending request to this free slot ---
                    pending_req & req = pending[qi];
                    const int64_t now = ggml_time_us();

                    c.seq_id          = req.seq_id;
                    c.priority        = req.priority;
                    c.t_queued        = req.t_queued;
                    c.t_first_sched   = now;
                    c.t_preempt_total = 0;
                    c.t_preempt_start = 0;
                    c.n_preemptions   = 0;
                    c.park_seq        = -1;
                    c.t_start_prompt  = now;
                    c.t_start_gen     = 0;
                    c.input           = req.input;
                    c.prompt          = req.prompt;
                    c.response        = "";

                    common_sampler_reset(c.smpl);

                    std::vector<llama_token> tokens_prompt;
                    tokens_prompt = common_tokenize(ctx, c.prompt, false);

                    for (size_t i = 0; i < tokens_prompt.size(); ++i) {
                        common_batch_add(batch, tokens_prompt[i], i + n_tokens_system, { c.id + 1 }, false);
                    }

                    if (batch.n_tokens > 0) {
                        batch.logits[batch.n_tokens - 1] = true;
                    }

                    c.n_prompt  = tokens_prompt.size();
                    c.n_decoded = 0;
                    c.i_batch   = batch.n_tokens - 1;

                    LOG_INF("\033[31m[t=%6.2f s] Client %3d: STARTED   seq %4d (priority %d) — "
                            "waited %.2f s in queue\033[0m\n",
                            (now - t_arrivals_base) / 1e6,
                            c.id, c.seq_id, c.priority,
                            (now - c.t_queued) / 1e6);

                    pending.erase(pending.begin() + qi);
                }
            }

            while (!pending.empty() && !park_ids_free.empty()) {
                const int qi = best_idx(pending);
                const int wi = worst_active_idx();

                if (wi == -1) break;
                if (pending[qi].priority >= clients[wi].priority) break;

                client & victim = clients[wi];
                pending_req & req = pending[qi];
                const int64_t now = ggml_time_us();

                LOG_INF("\033[35m[t=%6.2f s] Client %3d: PREEMPTED seq %4d (priority %d) "
                        "← seq %4d (priority %d) needs the slot\033[0m\n",
                        (now - t_arrivals_base) / 1e6,
                        victim.id, victim.seq_id, victim.priority,
                        req.seq_id, req.priority);

                // Park victim's KV cache (includes system prompt + full history)
                victim.park_seq = park_ids_free.back();
                park_ids_free.pop_back();

                const int32_t n_kv_preempt = n_tokens_system + victim.n_prompt + victim.n_decoded;
                {
                    const int64_t t_kv0 = ggml_time_us();
                    llama_kv_self_seq_cp(ctx, victim.id + 1, victim.park_seq, -1, -1);
                    // Reset the active slot KV to system prompt only
                    llama_kv_self_seq_rm(ctx, victim.id + 1, -1, -1);
                    llama_kv_self_seq_cp(ctx, 0, victim.id + 1, -1, -1);
                    LOG_INF("[t=%6.2f s] CTX_SWITCH type=preempt   n_tokens=%5d elapsed_us=%d\n",
                            (now - t_arrivals_base) / 1e6, n_kv_preempt,
                            (int)(ggml_time_us() - t_kv0));
                }

                victim.t_preempt_start = now;
                victim.n_preemptions++;

                // Move victim to the preempted list; smpl stays with the slot
                client saved_victim  = victim;
                saved_victim.smpl    = nullptr;
                saved_victim.i_batch = -1;
                preempted.push_back(std::move(saved_victim));

                // Assign the pending request to the freed slot
                victim.seq_id          = req.seq_id;
                victim.priority        = req.priority;
                victim.t_queued        = req.t_queued;
                victim.t_first_sched   = now;
                victim.t_preempt_total = 0;
                victim.t_preempt_start = 0;
                victim.n_preemptions   = 0;
                victim.park_seq        = -1;
                victim.t_start_prompt  = now;
                victim.t_start_gen     = 0;
                victim.input           = req.input;
                victim.prompt          = req.prompt;
                victim.response        = "";

                common_sampler_reset(victim.smpl);

                std::vector<llama_token> tokens_prompt;
                tokens_prompt = common_tokenize(ctx, victim.prompt, false);

                for (size_t i = 0; i < tokens_prompt.size(); ++i) {
                    common_batch_add(batch, tokens_prompt[i], i + n_tokens_system, { victim.id + 1 }, false);
                }

                if (batch.n_tokens > 0) {
                    batch.logits[batch.n_tokens - 1] = true;
                }

                victim.n_prompt  = tokens_prompt.size();
                victim.n_decoded = 0;
                victim.i_batch   = batch.n_tokens - 1;

                LOG_INF("\033[31mClient %3d: STARTED  seq %4d (priority %d) — "
                        "waited %.2f s in queue\033[0m\n",
                        victim.id, victim.seq_id, victim.priority,
                        (now - victim.t_queued) / 1e6);

                pending.erase(pending.begin() + qi);
            }
        }

        for (auto & c : clients) {
            if (c.seq_id == -1 || c.i_batch != -1) {
                continue;
            }

            c.i_batch = batch.n_tokens;

            common_batch_add(batch, c.sampled, n_tokens_system + c.n_prompt + c.n_decoded, { c.id + 1 }, true);

            c.n_decoded += 1;
        }

        if (batch.n_tokens == 0) {
            // Nothing to decode right now.
            if (future_reqs.empty() && pending.empty() && preempted.empty()) {
                break; // all done
            }
            // More requests will arrive later — spin until next arrival
            continue;
        }

        // process in chunks of params.n_batch
        int32_t n_batch = params.n_batch;

        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    LOG_ERR("%s : failed to decode the batch, n_batch = %d, ret = %d\n", __func__, n_batch, ret);
                    return 1;
                }

                LOG_ERR("%s : failed to decode the batch, retrying with n_batch = %d\n", __func__, n_batch / 2);

                n_cache_miss += 1;

                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;

                continue;
            }

            LOG_DBG("%s : decoded batch of %d tokens\n", __func__, n_tokens);

            for (auto & c : clients) {
                if (c.i_batch < (int) i || c.i_batch >= (int) (i + n_tokens)) {
                    continue;
                }

                const llama_token id = common_sampler_sample(c.smpl, ctx, c.i_batch - i);

                common_sampler_accept(c.smpl, id, true);

                if (c.n_decoded == 1) {
                    // start measuring generation time after the first token to make sure all concurrent clients
                    // have their prompt already processed
                    c.t_start_gen = ggml_time_us();
                }

                const std::string token_str = common_token_to_piece(ctx, id);

                c.response += token_str;
                c.sampled = id;

                if (c.n_decoded > 2 &&
                        (llama_vocab_is_eog(vocab, id) ||
                         (params.n_predict > 0 && c.n_decoded + c.n_prompt >= params.n_predict) ||
                         c.response.find("User:") != std::string::npos ||
                         c.response.find('\n') != std::string::npos)) {
                    // basic reverse prompt
                    const size_t pos = c.response.find("User:");
                    if (pos != std::string::npos) {
                        c.response = c.response.substr(0, pos);
                    }

                    // delete only the generated part of the sequence, i.e. keep the system prompt in the cache
                    const int32_t n_kv_complete = n_tokens_system + c.n_prompt + c.n_decoded;
                    {
                        const int64_t t_kv0 = ggml_time_us();
                        llama_kv_self_seq_rm(ctx,    c.id + 1, -1, -1);
                        llama_kv_self_seq_cp(ctx, 0, c.id + 1, -1, -1);
                        const int64_t t_kv1 = ggml_time_us();
                        LOG_INF("[t=%6.2f s] CTX_SWITCH type=complete  n_tokens=%5d elapsed_us=%d\n",
                                (t_kv1 - t_arrivals_base) / 1e6, n_kv_complete,
                                (int)(t_kv1 - t_kv0));
                    }

                    const auto t_main_end = ggml_time_us();

                    const double t_queue_wait_s = (c.t_first_sched - c.t_queued)   / 1e6;
                    const double t_preempted_s  =  c.t_preempt_total               / 1e6;

                    LOG_INF("\033[31m[t=%6.2f s] Client %3d: DONE      seq %3d/%3d (priority %d) — "
                            "prompt %4d t, gen %4d t, total %5.2f s, speed %5.2f t/s, "
                            "wait %5.2f s, preempted %dx %.2f s\033[0m\n",
                            (t_main_end - t_arrivals_base) / 1e6,
                            c.id, c.seq_id, n_seq, c.priority,
                            c.n_prompt, c.n_decoded,
                            (t_main_end - c.t_start_prompt) / 1e6,
                            (double)(c.n_prompt + c.n_decoded) / (t_main_end - c.t_start_prompt) * 1e6,
                            t_queue_wait_s,
                            c.n_preemptions, t_preempted_s);

                    n_total_prompt += c.n_prompt;
                    n_total_gen    += c.n_decoded;

                    c.seq_id = -1;
                }

                c.i_batch = -1;
            }
        }
    }

    const auto t_main_end = ggml_time_us();

    print_date_time();

    LOG_INF("%s: n_parallel = %d, n_sequences = %d, cont_batching = %d, system tokens = %d\n",
            __func__, n_clients, n_seq, cont_batching, n_tokens_system);
    if (params.prompt_file.empty()) {
        params.prompt_file = "used built-in defaults";
    }
    LOG_INF("External prompt file: \033[32m%s\033[0m\n", params.prompt_file.c_str());
    LOG_INF("Model and path used:  \033[32m%s\033[0m\n\n", params.model.c_str());

    LOG_INF("Total prompt tokens: %6d, speed: %5.2f t/s\n", n_total_prompt, (double) (n_total_prompt              ) / (t_main_end - t_main_start) * 1e6);
    LOG_INF("Total gen tokens:    %6d, speed: %5.2f t/s\n", n_total_gen,    (double) (n_total_gen                 ) / (t_main_end - t_main_start) * 1e6);
    LOG_INF("Total speed (AVG):   %6s  speed: %5.2f t/s\n", "",             (double) (n_total_prompt + n_total_gen) / (t_main_end - t_main_start) * 1e6);
    LOG_INF("Cache misses:        %6d\n", n_cache_miss);

    LOG_INF("\n");

    // TODO: print sampling/grammar timings for all clients
    llama_perf_context_print(ctx);

    llama_batch_free(batch);

    llama_init.context.reset();
    llama_init.model.reset();

    llama_backend_free();

    LOG("\n\n");

    common_log_pause(common_log_main());
    common_log_set_file(common_log_main(), nullptr);

    exit(0);
}
