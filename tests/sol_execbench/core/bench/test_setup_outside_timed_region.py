# SPDX-License-Identifier: Apache-2.0
"""Per-iteration setup must never be counted in the reported latency.

`bench_time_with_cuda_events` calls `setup()` — which in the real harness is
`ShiftingMemoryPoolAllocator.get_unique_args`, copying a fresh copy of every
input into a new pool offset — once per iteration, immediately before the timed
region. Its docstring promises "Setup time is **not** included in measurements."

If that ever stopped being true, every latency this benchmark has ever produced
on any part would be inflated by an allocator copy, `T_b` and `T_k` alike, and
nothing in any artifact would say so. The question was asked during the MI355X
clock-bracket work and the answer was "no, it is outside" — this test is what
keeps the answer true, because reading the code proves it today and a test
proves it tomorrow.

The check is a direct causal one: make `setup` sleep, and require the reported
latency not to move. A 50 ms sleep against a sub-millisecond kernel is a signal
roughly 100x the noise, so there is no ambiguity about which way it came out.
"""

from __future__ import annotations

import statistics
import time

import pytest

torch = pytest.importorskip("torch")

from sol_execbench.core.bench.timing import (  # noqa: E402
    bench_time_with_cuda_events,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU to record timing events"
)

SLEEP_S = 0.050


def _median_latency(sleep_s: float) -> float:
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")

    def setup():
        if sleep_s:
            time.sleep(sleep_s)
        return (a, b)

    times = bench_time_with_cuda_events(
        fn=lambda args: torch.mm(*args), warmup=5, rep=20, setup=setup,
        device="cuda:0",
    )
    return statistics.median(times)


def test_a_slow_setup_does_not_inflate_the_reported_latency():
    fast = _median_latency(0.0)
    slow = _median_latency(SLEEP_S)
    delta_ms = slow - fast
    assert delta_ms < SLEEP_S * 1000 * 0.1, (
        f"setup appears to be INSIDE the timed region: a {SLEEP_S * 1000:.0f} ms "
        f"sleep in setup moved the reported median by {delta_ms:.3f} ms "
        f"({fast:.4f} -> {slow:.4f}). Every latency this benchmark has produced "
        f"would include per-iteration allocator work."
    )


def test_the_event_pair_encloses_only_the_function_call():
    """The structural half of the same property, so a failure says WHERE.

    `setup()`, the L2 flush and `_fence_streams()` all precede
    `start_events[i].record()`; only `fn(args)` and `_join_streams()` sit between
    the two events.
    """
    import inspect

    src = inspect.getsource(bench_time_with_cuda_events)
    body = src.split("for i in range(rep):")[1]
    timed = body.split("start_events[i].record()")[1].split("end_events[i].record()")[0]
    assert "setup()" not in timed
    assert "_clear_cache" not in timed
    assert "_reset_persisting_l2_cache" not in timed
    assert "fn(args)" in timed


# ---------------------------------------------------------------------------
# The same invariant, for the pre-window settle.
#
# The settle runs the real kernel until the card's clock stops climbing. It is
# GPU work added immediately before timing, so the question "does it leak into
# the measurement?" has to be answered the same causal way, or the settle would
# be indistinguishable from inflating every latency by a second of warm-up.
# ---------------------------------------------------------------------------


def test_the_settle_runs_before_the_window_and_not_inside_it():
    """`bracketed(settle=...)` must not change the number the thunk returns.

    Injecting a sleep into the settle stands in for an arbitrarily expensive
    settle: however long it takes, the latency the thunk reports is the thunk's
    alone.
    """
    from sol_execbench.core.bench.clock_bracket import bracketed

    def thunk():
        return _median_latency(0.0)

    plain, _ = bracketed(thunk, device=0, sampler=lambda _d: 2000,
                         settle=None)
    settled, br = bracketed(
        thunk, device=0, sampler=lambda _d: 2000,
        settle=lambda: time.sleep(0.010), window_iters=1,
    )
    assert br.settle is not None and br.settle["settle_iterations"] > 0
    # Same measurement, within the run-to-run noise of a 512x512 mm.
    assert abs(settled - plain) < SLEEP_S * 1000 * 0.1, (
        f"the settle appears to be inside the measurement: {plain:.4f} -> "
        f"{settled:.4f} ms"
    )


def test_the_settle_precedes_the_first_clock_sample():
    """Ordering is the whole design. A settle that ran after the `before`
    sample, or inside the window, would leave the bracket measuring the ramp it
    exists to remove."""
    from sol_execbench.core.bench.clock_bracket import bracketed

    events = []
    clocks = iter([2000] * 50)

    def sampler(_d):
        events.append("sample")
        return next(clocks)

    bracketed(lambda: events.append("thunk"), device=0, sampler=sampler,
              settle=lambda: events.append("settle"), window_iters=1)

    # `settle_clock` takes its own entry sample first, so the very first event is
    # a sample; what must hold is that ALL settle work is finished, and a fresh
    # sample taken, before the thunk runs -- and another sample after it.
    i_thunk = events.index("thunk")
    last_settle = len(events) - 1 - events[::-1].index("settle")
    assert last_settle < i_thunk, "settle work leaked into or past the window"
    assert "sample" in events[last_settle:i_thunk], \
        "the `before` sample must be taken after the settle, not before it"
    assert "sample" in events[i_thunk:], "no `after` sample"
    assert events[-1] == "sample"
