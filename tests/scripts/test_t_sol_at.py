# SPDX-License-Identifier: Apache-2.0
"""T_SOL must be evaluatable at the clock a measurement actually ran at.

The property that matters most is the first one: evaluating at the reference clock
has to reproduce what `sol_bounds.py` already wrote, bit for bit. Without that,
switching to per-measurement clocks would silently move every existing score and
there would be no way to tell that drift from the intended change.
"""

from __future__ import annotations

import math

import pytest

from solexbench_rocm.t_sol_at import (
    MissingBoundTerms,
    bottleneck_at,
    t_sol_cycles_at,
    t_sol_ms_at,
)

F_REF = 1650.0
DRAM_BPS = 8.0e12          # 8 TB/s, as in SOLAR/configs/arch/MI355X.yaml


def _record(compute_cycles: float, memory_bytes: int) -> dict:
    return {
        "compute_cycles": compute_cycles,
        "memory_bytes": memory_bytes,
        "dram_byte_per_sec": DRAM_BPS,
    }


def _sol_bounds_reference(w: dict, f_mhz: float) -> int:
    """Recompute the way scripts/sol_bounds.py does, from freq-derived per-cycle."""
    dram_byte_per_cycle = w["dram_byte_per_sec"] / (f_mhz * 1e6)
    return max(1, math.ceil(max(
        w["compute_cycles"],
        w["memory_bytes"] / dram_byte_per_cycle,
    )))


@pytest.mark.parametrize("compute_cycles,memory_bytes", [
    (31142.6, 150994944),      # memory-bound, from a real L1 record
    (900000.0, 150994944),     # compute-bound
    (1.0, 12288),              # sub-cycle small end, where the ceil floor bites
    (0.0, 4096),               # no MACs at all
])
def test_matches_sol_bounds_at_reference_clock(compute_cycles, memory_bytes):
    """At F_ref the rescale must agree exactly with the deriver's own arithmetic."""
    w = _record(compute_cycles, memory_bytes)
    assert t_sol_cycles_at(w, F_REF) == _sol_bounds_reference(w, F_REF)


def test_compute_bound_time_scales_inversely_with_clock():
    """A compute-bound bound is a fixed cycle count, so 2x the clock halves the time."""
    w = _record(compute_cycles=1e7, memory_bytes=1024)     # tiny bytes -> compute wins
    assert bottleneck_at(w, F_REF) == "compute"
    t_slow = t_sol_ms_at(w, 1650.0)
    t_fast = t_sol_ms_at(w, 3300.0)
    assert t_fast == pytest.approx(t_slow / 2, rel=1e-9)


def test_memory_bound_time_does_not_move_with_clock():
    """HBM does not run off the core clock, so boosting must not tighten the bound.

    This is the case that a naive "cycles / new_F" rescale gets wrong, and it is the
    majority of the corpus -- 1135 of 2689 derived workloads are memory-bound.
    """
    w = _record(compute_cycles=1.0, memory_bytes=8_000_000)   # bytes dominate
    assert bottleneck_at(w, F_REF) == "memory"
    t_ref = t_sol_ms_at(w, F_REF)
    for f in (1730.0, 1837.0, 2394.0):
        assert bottleneck_at(w, f) == "memory"
        assert t_sol_ms_at(w, f) == pytest.approx(t_ref, rel=1e-6)


def test_bottleneck_can_flip_as_the_clock_rises():
    """Compute-bound at 1650 can become memory-bound at 2394, and the max must follow.

    Scaling whichever term won at the reference clock would miss this and understate
    the bound. Sized so the crossover sits between the two clocks: compute_cycles is
    chosen to be just above memory_cycles(1650) and below memory_cycles(2394).
    """
    memory_bytes = 8_000_000
    mem_cycles_at = lambda f: memory_bytes * f * 1e6 / DRAM_BPS
    compute_cycles = (mem_cycles_at(1650.0) + mem_cycles_at(2394.0)) / 2
    w = _record(compute_cycles, memory_bytes)

    assert bottleneck_at(w, 1650.0) == "compute"
    assert bottleneck_at(w, 2394.0) == "memory"

    # Crossing into the memory regime, time must stop falling with 1/F and flatten.
    t_lo = t_sol_ms_at(w, 1650.0)
    t_hi = t_sol_ms_at(w, 2394.0)
    naive = t_lo * 1650.0 / 2394.0          # what pure 1/F scaling would predict
    assert t_hi > naive, "bound must not be tightened past the memory floor"
    assert t_hi == pytest.approx(memory_bytes / DRAM_BPS * 1e3, rel=1e-6)


def test_higher_clock_never_loosens_the_bound():
    """Monotonicity: more clock can only make the speed of light faster or equal."""
    for compute_cycles, memory_bytes in [(1e6, 1e6), (1e3, 8e6), (1e8, 12288)]:
        w = _record(float(compute_cycles), int(memory_bytes))
        ts = [t_sol_ms_at(w, f) for f in (1650, 1730, 1837, 2000, 2394)]
        assert all(b <= a + 1e-12 for a, b in zip(ts, ts[1:])), ts


def test_old_records_are_refused_not_guessed():
    """A record with only the max of the two terms cannot be rescaled -- say so.

    Guessing from `bottleneck` would happen to work while F only ever rises above
    F_ref, and would silently produce wrong bounds the first time it did not.
    """
    old = {"t_sol_cycles_exact": 31142.6, "bottleneck": "memory",
           "memory_bytes": 150994944}
    with pytest.raises(MissingBoundTerms, match="sol_bounds"):
        t_sol_ms_at(old, 1730.0)


def test_rejects_nonsense_clock():
    w = _record(1e6, 1_000_000)
    for bad in (0.0, -1650.0):
        with pytest.raises(ValueError):
            t_sol_ms_at(w, bad)
