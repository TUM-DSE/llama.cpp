#pragma once

// Guardian TTFT instrumentation for the HTTP server.
//
// The benchmark measures four points around one request by writing to I/O port
// 0xf4, which the host traces with bpftrace on the kvm_pio tracepoint:
//
//   200  agent sends the request        (agents/agent_simple_infer.py)
//   201  engine first handles it        (here)
//   202  engine posts the response      (here)
//   203  agent receives it              (agents/agent_simple_infer.py)
//
// The agent tags every request with an `outb_id` so the four events can be
// paired per request; the value written is `(id << 16) | event`, which is what
// benchmarks/ttft/ttft_eval.bt decodes.
//
// This is deliberately a self-contained copy of `native/`'s `my_outl` rather
// than a dependency on libguardian: the HTTP server is the *baseline* for the
// shared-memory transport, and linking the shm library into it would blur the
// thing the benchmark compares. The shm server (tools/guardian-server) emits
// the same two events from its own path.
//
// History: this existed in the legacy fork's examples/server/server.cpp
// (commits 949af8f1d, 8d7a5affe) and was not carried across the modern
// llama.cpp migration, so vm/cvm TTFT runs captured only the agent-side pair
// (100/0/0/100) while guardian captured all four.

#include <cstdint>

#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__))
#    define GUARDIAN_INSTR_SUPPORTED 1
#    include <sys/io.h>
#else
#    define GUARDIAN_INSTR_SUPPORTED 0
#endif

#define GUARDIAN_BENCHMARK_PORT 0xf4
#define GUARDIAN_EVENT_ENGINE_RECV 201
#define GUARDIAN_EVENT_ENGINE_SEND 202

// Logged once at startup so a benchmark can tell an instrumented binary from an
// uninstrumented one *before* spending a hardware run on it.
#define GUARDIAN_INSTR_MARKER "guardian: TTFT port instrumentation active"

// Port access is granted per thread, so this is requested once per thread and
// the result cached. A denial (no CAP_SYS_RAWIO) disables instrumentation for
// that thread instead of faulting: an uninstrumented run is a missing
// measurement, a SIGSEGV in the server is a lost one.
inline bool guardian_instr_ready() {
#if GUARDIAN_INSTR_SUPPORTED
    static thread_local int state = -1;
    if (state < 0) {
        state = ioperm(GUARDIAN_BENCHMARK_PORT, 4, 1) == 0 ? 1 : 0;
    }
    return state == 1;
#else
    return false;
#endif
}

// `outb_id` is -1 unless the caller asked to be traced, so untraced traffic
// costs one comparison.
inline void guardian_port_event(int outb_id, unsigned int event) {
#if GUARDIAN_INSTR_SUPPORTED
    if (outb_id < 0 || !guardian_instr_ready()) {
        return;
    }
    const uint32_t value = ((uint32_t) (outb_id & 0xFFFF) << 16) | (event & 0xFFFF);
    outl(value, GUARDIAN_BENCHMARK_PORT);
#else
    (void) outb_id;
    (void) event;
#endif
}
