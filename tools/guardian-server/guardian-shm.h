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
#include <memory>
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

    std::unique_ptr<Communication> comm;
    server_context * ctx_server = nullptr;
    const common_params * params_base = nullptr;
    std::unique_ptr<server_context_meta> meta;
    const llama_vocab * vocab = nullptr;
    std::atomic<bool> running{false};
    std::thread scanner;
};
