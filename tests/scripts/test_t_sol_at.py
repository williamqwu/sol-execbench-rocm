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
    INTERVAL_FIELDS,
    MissingBoundTerms,
    bottleneck_at,
    t_sol_cycles_at,
    t_sol_interval,
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


# ---------------------------------------------------------------------------
# T_SOL as an interval over a clock bracket.
#
# The clock moves inside the timed window on this part, so a single T_SOL is a
# claim the measurement does not support. These pin the three things the interval
# has to get right: which end is published, how wide it is, and what happens to
# the bottleneck across it.
# ---------------------------------------------------------------------------

#: The two problems the methodology decision was argued over, measured on
#: artifacts/06-MI355X. WIDE moves the compute term by 33.7%; CLEAN by 3.3%.
WIDE_BRACKET = (1607.0, 2148.0)
CLEAN_BRACKET = (2314.0, 2390.0)


def test_t_sol_is_larger_at_the_minimum_clock():
    """Less clock, more milliseconds. The whole reason the published end is f_min."""
    w = _record(compute_cycles=1e7, memory_bytes=1024)
    iv = t_sol_interval(w, *WIDE_BRACKET)
    assert iv["t_sol_ms_at_clock_min"] >= iv["t_sol_ms_at_clock_max"]


def test_the_published_value_is_the_minimum_clock_end():
    """Published = tightest = largest T_SOL = evaluated at f_min.

    If this ever flips to f_max the bound becomes one nothing can violate, and the
    "no measurement beats its T_SOL" check -- the only thing that can catch a wrong
    bound from the outside -- goes permanently quiet. That is the failure mode
    CLAUDE.md §6 names, so it gets an explicit test rather than a comment.
    """
    w = _record(compute_cycles=1e7, memory_bytes=1024)
    iv = t_sol_interval(w, *WIDE_BRACKET)
    assert iv["t_sol_ms_published"] == iv["t_sol_ms_at_clock_min"]
    assert iv["t_sol_ms_published"] == t_sol_ms_at(w, min(WIDE_BRACKET))
    assert iv["t_sol_published_at_mhz"] == min(WIDE_BRACKET)
    assert iv["t_sol_published_end"] == "clock_min"
    # ...and it really is the larger of the two, not merely the one so labelled.
    assert iv["t_sol_ms_published"] > iv["t_sol_ms_at_clock_max"]


def test_the_bracket_order_does_not_decide_the_published_end():
    """`before` may be the higher sample; the bound must not depend on which."""
    w = _record(compute_cycles=1e7, memory_bytes=1024)
    lo_first = t_sol_interval(w, 1607.0, 2148.0)
    hi_first = t_sol_interval(w, 2148.0, 1607.0)
    assert lo_first == hi_first


def test_interval_collapses_to_a_point_when_the_bracket_does():
    """A card that read the same clock twice has no interval, and must say 0.0.

    Zero width here is a measurement -- "the clock did not move" -- and must be
    distinguishable from a missing interval, which is None.
    """
    w = _record(compute_cycles=1e7, memory_bytes=1024)
    iv = t_sol_interval(w, 2390.0, 2390.0)
    assert iv["t_sol_ms_at_clock_min"] == iv["t_sol_ms_at_clock_max"]
    assert iv["t_sol_interval_width_rel"] == 0.0
    assert iv["t_sol_interval_halfwidth_rel"] == 0.0
    assert iv["t_sol_bottleneck_flips"] is False


def test_a_memory_bound_workload_has_a_zero_width_interval():
    """The correctness check on the whole idea.

    The memory term is a fixed TIME -- bytes over bytes-per-second, with no clock
    in it -- so a memory-bound bound does not move however wide the bracket is. If
    this ever reads non-zero, the interval is being computed by scaling cycles by
    1/F somewhere, which is the exact error the split into two terms exists to
    prevent, and every compute-bound width would be wrong too.
    """
    w = _record(compute_cycles=1.0, memory_bytes=8_000_000)
    iv = t_sol_interval(w, *WIDE_BRACKET)          # a 33.7% clock span
    assert iv["t_sol_bottleneck_at_clock_min"] == "memory"
    assert iv["t_sol_bottleneck_at_clock_max"] == "memory"
    assert iv["t_sol_interval_halfwidth_rel"] == pytest.approx(0.0, abs=1e-12)
    assert iv["t_sol_ms_published"] == pytest.approx(
        iv["t_sol_ms_at_clock_max"], rel=1e-12)


def test_a_bottleneck_flip_across_the_interval_is_reported():
    """Compute-bound at one end, memory-bound at the other: say so, do not pick one.

    Across the 33-43% spans measured on this corpus this really happens, and a
    record naming a single bottleneck would be asserting something the bracket
    does not support.
    """
    memory_bytes = 8_000_000
    mem_cycles_at = lambda f: memory_bytes * f * 1e6 / DRAM_BPS
    compute_cycles = (mem_cycles_at(1607.0) + mem_cycles_at(2148.0)) / 2
    iv = t_sol_interval(_record(compute_cycles, memory_bytes), *WIDE_BRACKET)
    assert iv["t_sol_bottleneck_at_clock_min"] == "compute"
    assert iv["t_sol_bottleneck_at_clock_max"] == "memory"
    assert iv["t_sol_bottleneck_flips"] is True


def test_width_separates_a_wide_bracket_from_a_clean_one():
    """The field exists to be sorted on, so the ordering it produces is the test."""
    w = _record(compute_cycles=1e7, memory_bytes=1024)      # compute-bound
    wide = t_sol_interval(w, *WIDE_BRACKET)
    clean = t_sol_interval(w, *CLEAN_BRACKET)
    # The compute term scales as 1/F, so the span of T_SOL is the span of the
    # clock: 2148/1607 = 1.337 and 2390/2314 = 1.033, the figures the decision
    # was argued from.
    assert wide["t_sol_interval_width_rel"] == pytest.approx(
        2148.0 / 1607.0 - 1, rel=1e-3)
    assert clean["t_sol_interval_width_rel"] == pytest.approx(
        2390.0 / 2314.0 - 1, rel=1e-3)
    # +-14.4% against +-1.6%: an order of magnitude apart, which is the whole
    # reason a reader needs the width beside the number.
    assert wide["t_sol_interval_halfwidth_rel"] == pytest.approx(0.144, abs=5e-3)
    assert clean["t_sol_interval_halfwidth_rel"] == pytest.approx(0.016, abs=5e-3)


def test_every_declared_interval_field_is_emitted():
    """One list of field names, so the manifest and the scorer cannot drift."""
    iv = t_sol_interval(_record(1e7, 1024), *WIDE_BRACKET)
    assert set(iv) == set(INTERVAL_FIELDS)


def test_interval_refuses_a_record_it_cannot_re_clock():
    old = {"t_sol_cycles_exact": 31142.6, "bottleneck": "memory"}
    with pytest.raises(MissingBoundTerms):
        t_sol_interval(old, *WIDE_BRACKET)
