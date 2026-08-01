#pragma once

// Guardian: where a request's wall-clock time goes before prefill starts.
//
// Two intervals, both reported in the response timings (`queued_time_ms` /
// `deferred_time_ms`): time sitting in the ready queue, and time parked in the
// deferred queue because no slot was free. Reporting them is what makes an
// engine-side latency attributable instead of merely visible.
//
// Each clock carries an explicit open flag, and banking an interval clears it.
// That is what makes closing idempotent, and it matters: a task popped from
// the ready queue is banked at the pop, and if no slot turns out to be free it
// is then deferred, which would otherwise bank the same interval a second
// time. The legacy fork (792b07294) had exactly that double count. The flag is
// a flag rather than a zero start stamp so that a caller supplying a clock
// based at zero still measures.
//
// The state machine takes `now_us` from its caller rather than reading a clock,
// so the transitions can be pinned deterministically without a server, a model
// or a real clock — see the guardian repo's native/tests/queue_timing_test.cpp.
// server-queue.cpp is the only production caller and supplies ggml_time_us().
//
// WHERE THESE INTERVALS SIT RELATIVE TO THE TTFT PORT EVENTS, which is not the
// same on both transports and is why benchmarks/ttft/bench.py reports them as
// server diagnostics rather than as a split of the traced legs:
//   * HTTP: the clock is opened in server_queue::post() on the HTTP thread and
//     its first interval is banked when start_loop() pops the task, which is
//     immediately before process_single_task() emits event 201. That first wait
//     is therefore inside `201 - 200`, the agent->engine leg.
//   * shm: tools/guardian-server/guardian-shm.cpp emits its own 201 before
//     post_task(), so the same interval falls inside `202 - 201`.
// Deferred time, and any ready-queue time after a re-dispatch, are inside
// `202 - 201` on both.

// STAGE STAMPS (the 201→202 accounting, PLAN_REMAINING Phase 10). The queue
// intervals above never explained the measured round trips (ready-queue
// 0.1–0.2 ms, slot-wait 0, yet 34–46 ms of `202 - 201` unattributed on
// 2026-07-29), so the clock also records three wall-clock points:
//
//   t_recv_us        the transport's event 201 — request parsed, no
//                    per-request work done yet. Set by the transport
//                    (handle_completions_impl() / guardian-shm.cpp), because
//                    only it knows when 201 was emitted.
//   t_post_us        first entry into the ready queue. `prep` (tokenization,
//                    schema evaluation, task construction) is
//                    t_post - t_recv.
//   t_last_leave_us  the most recent departure from either queue. `launch`
//                    (slot selection, prompt setup, batch admission) is
//                    t_start_process_prompt - t_last_leave.
//
// Stamps rather than banked intervals: the spans they delimit end at points
// this struct never sees (prompt start lives on the slot, 202 on the result),
// so the subtraction has to happen where both ends are visible
// (server_slot::get_timings(), result_timings::to_json()).

#include <cstdint>

struct guardian_queue_clock {
    bool    queued_open       = false;
    int64_t queued_us_start   = 0;
    double  queued_ms         = 0.0;
    bool    deferred_open     = false;
    int64_t deferred_us_start = 0;
    double  deferred_ms       = 0.0;

    int64_t t_recv_us       = 0;  // event 201; set by the transport
    int64_t t_post_us       = 0;  // first open_queued()
    int64_t t_last_leave_us = 0;  // latest bank_queued()/bank_deferred()

    // entering the ready queue: on post, or on coming back from deferred
    void open_queued(int64_t now_us) {
        queued_open     = true;
        queued_us_start = now_us;
        if (t_post_us == 0) {
            t_post_us = now_us;
        }
    }

    // leaving the ready queue; a no-op if the clock is already closed
    void bank_queued(int64_t now_us) {
        if (queued_open) {
            queued_ms  += (now_us - queued_us_start) / 1000.0;
            queued_open = false;
            t_last_leave_us = now_us;
        }
    }

    void open_deferred(int64_t now_us) {
        deferred_open     = true;
        deferred_us_start = now_us;
    }

    void bank_deferred(int64_t now_us) {
        if (deferred_open) {
            deferred_ms  += (now_us - deferred_us_start) / 1000.0;
            deferred_open = false;
            t_last_leave_us = now_us;
        }
    }

    // no slot was free: close the ready-queue clock (the pop already banked it,
    // so this is the no-op branch above) and start the deferred one
    void defer(int64_t now_us) {
        bank_queued(now_us);
        open_deferred(now_us);
    }

    // a slot freed up: bank the deferred time and rejoin the ready queue
    void undefer(int64_t now_us) {
        bank_deferred(now_us);
        open_queued(now_us);
    }
};
