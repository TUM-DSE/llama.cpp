#pragma once

// Guardian shared-memory front-end for the llama.cpp server core.
//
// Serves inference requests arriving over the Guardian shm protocol
// (native/include/guardian/shm.h in the guardian repo: 512 Request slots in a
// POSIX shm region, semaphore handshake) by translating them into server_task
// completions on the same task queue the HTTP front-end uses. Reimplementation
// of the legacy examples/guardian_server (old fork, branch guardian-legacy)
// against the modularized server-context API.

#include "server-context.h"

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

struct common_params;
struct Communication;

struct guardian_shm_frontend {
    guardian_shm_frontend();
    ~guardian_shm_frontend();

    // create + initialize the shm region (server side, create=true)
    bool init();

    // start the scanner thread; requires a loaded model (reads vocab/meta).
    // must be called after ctx.load_model() and before ctx.start_loop().
    void start(server_context & ctx, const common_params & params);

    // signal shutdown and join the scanner (detached request handlers finish
    // on their own; pending readers are unblocked via the should_stop poll)
    void stop();

private:
    void scan_loop();
    void handle_request(int slot_idx);

    // write the response back into the slot, retire it from the in-flight
    // table and post the client's semaphore. `outcome` is a short label for
    // the completion log (ok / llama-timeout / invalid-json / ...).
    void finish_request(int slot_idx, const std::string & body, bool io_ok,
                        int outb_id, const char * outcome);

    // one shm slot currently being served, for the stall watchdog below
    struct inflight_entry {
        int  req_id  = -1;
        int  outb_id = -1;
        int  task_id = -1; // llama.cpp task id, -1 until the task is posted
        std::chrono::steady_clock::time_point started;
    };

    void inflight_begin(int slot_idx, int req_id, int outb_id);
    void inflight_set_task(int slot_idx, int task_id);

    // log every slot in flight longer than GUARDIAN_STALL_WARN_SECONDS, plus
    // the pending serverNotifier counts — a request that the scanner never
    // dispatched shows up here and nowhere else
    void report_stalls();

    std::unique_ptr<Communication> comm;
    server_context * ctx_server = nullptr;
    const common_params * params_base = nullptr;
    std::unique_ptr<server_context_meta> meta;
    const llama_vocab * vocab = nullptr;
    std::atomic<bool> running{false};
    std::thread scanner;

    std::mutex inflight_mutex;
    std::map<int, inflight_entry> inflight;
    // scanner-thread only (no lock): when each still-posted serverNotifier was
    // first seen, so a genuinely undispatched request can be told apart from
    // one posted moments ago
    std::map<int, std::chrono::steady_clock::time_point> notifier_pending_since;
    std::chrono::steady_clock::time_point last_stall_report{};
};
