# SPDX-License-Identifier: Apache-2.0
"""T_SOL must be evaluatable at the clock a measurement actually ran at.

The property that matters most is the first one: evaluating at the reference clock
has to reproduce what `sol_bounds.py` already wrote, bit for bit. Without that,
switching to per-measurement clocks would silently move every existing score and
there would be no way to tell that drift from the intended change.
"""

from __future__ import annotations

import math
import sys

from pathlib import Path

import pytest

from solexbench_rocm.t_sol_at import (
    INTERVAL_FIELDS,
    MissingBoundTerms,
    MissingReferenceClock,
    REFERENCE_CLOCK_FIELD,
    bottleneck_at,
    bound_ms,
    reference_clock_mhz,
    t_sol_cycles_at,
    t_sol_interval,
    t_sol_ms_at,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import sol_bounds  # noqa: E402  -- the writer half of the same contract

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


# ---------------------------------------------------------------------------
# D63: a stored cycle count is only legible next to the clock it was expressed
# at, and nothing enforced that.
#
# `artifacts/03-MI355X/t_sol.json` holds 2902 records at 1.8 GHz and 96 at
# 2.4 GHz under a header declaring 2400, and two consumers read the column as
# if one clock described it. These pin both halves of the fix: the reader that
# refuses a record which does not say (`bound_ms`), and the writer rule that
# stops a header describing a body it did not compute.
# ---------------------------------------------------------------------------

#: A record in the shape `sol_bounds.py` writes today, stamped with its clock.
def _stamped(compute_cycles: float, memory_bytes: int, f_ref_mhz: float) -> dict:
    w = _record(compute_cycles, memory_bytes)
    w["f_ref_mhz"] = f_ref_mhz
    w["t_sol_cycles"] = t_sol_cycles_at(w, f_ref_mhz)
    w["t_sol_ms"] = w["t_sol_cycles"] / (f_ref_mhz * 1e3)
    return w


def test_bound_ms_refuses_a_record_that_names_no_clock():
    """The whole point. An unstamped ms column is a number with no unit."""
    legacy = _record(1e7, 1024)
    legacy["t_sol_cycles"], legacy["t_sol_ms"] = 10_000_000, 5.5555
    assert reference_clock_mhz(legacy) is None
    with pytest.raises(MissingReferenceClock, match="D63"):
        bound_ms(legacy)


@pytest.mark.parametrize("stamp", [None, 0, 0.0, -1800.0, "", "fast"])
def test_a_stamp_that_names_no_clock_is_absent_not_repaired(stamp):
    """`f_ref_mhz: 0` is as uninterpretable as no key at all -- refuse both."""
    w = _stamped(1e7, 1024, 1800.0)
    w["f_ref_mhz"] = stamp
    assert reference_clock_mhz(w) is None
    with pytest.raises(MissingReferenceClock):
        bound_ms(w)


def test_bound_ms_round_trips_the_clock_it_was_written_at():
    """A stamped record's stored ms must be its own bound at its own clock.

    This is the identity that makes the stamp worth having: with it, a reader
    can re-derive the column and check it; without it, the column can only be
    believed.
    """
    for f_ref in (1800.0, 2400.0):
        for compute_cycles, memory_bytes in [(1e7, 1024), (1.0, 8_000_000),
                                             (0.0, 4096), (31142.6, 150994944)]:
            w = _stamped(compute_cycles, memory_bytes, f_ref)
            assert bound_ms(w) == t_sol_ms_at(w, w["f_ref_mhz"])


def test_bound_ms_returns_the_stored_column_and_does_not_re_clock_it():
    """Reading is not converting. `bound_ms` hands back what the file says.

    And the file may well say 1.8 GHz: 2902 of 2998 MI355X records do. A reader
    that "helpfully" evaluated at the arch peak instead would shrink a
    compute-bound bound by exactly the 1.333x that is D63, in the undetectable
    direction.
    """
    at18 = _stamped(1e7, 1024, 1800.0)                 # compute-bound
    assert bound_ms(at18) == at18["t_sol_ms"]
    assert bound_ms(at18) == pytest.approx(
        t_sol_ms_at(at18, 2400.0) * (2400.0 / 1800.0), rel=1e-9)


def test_two_records_on_two_clocks_are_each_legible_and_do_not_average():
    """The mixed file, in miniature: same workload, two f_ref, both readable."""
    at18 = _stamped(1e7, 1024, 1800.0)
    at24 = _stamped(1e7, 1024, 2400.0)
    assert reference_clock_mhz(at18) == 1800.0
    assert reference_clock_mhz(at24) == 2400.0
    assert bound_ms(at18) / bound_ms(at24) == pytest.approx(4 / 3, rel=1e-9)
    # ...and the clock-free re-derivation agrees with both, which is why the
    # published bound never carried the defect.
    assert t_sol_ms_at(at18, 2000.0) == t_sol_ms_at(at24, 2000.0)


def test_bound_ms_distinguishes_no_clock_from_no_bound():
    """A stamped record with no `t_sol_ms` is not a bounded workload.

    Different failure, different exception: conflating them would let a caller
    "handle" a missing clock by skipping records that simply have no bound.
    """
    unbounded = {"f_ref_mhz": 2400.0, "error": "SOLAR stage 1 produced nothing"}
    with pytest.raises(KeyError) as e:
        bound_ms(unbounded)
    assert not isinstance(e.value, MissingReferenceClock)


def test_the_field_name_is_shared_with_the_writers():
    """One spelling, so the two tier writers and every reader cannot drift."""
    assert REFERENCE_CLOCK_FIELD == "f_ref_mhz"
    # Comparable for equality across writers -- `1.005 * 1000` is not.
    assert sol_bounds._mhz(2.4) == 2400.0
    assert sol_bounds._mhz(1.8) == 1800.0
    assert sol_bounds._mhz(1.005) == 1005.0


# ---- the writer side of the same contract: scripts/sol_bounds.py -----------

def _problem(*f_refs) -> dict:
    """A per-problem result whose bounded workloads carry *f_refs* in order.

    `None` stands for a record written before the field existed -- the shape
    every cached result on disk has today.
    """
    workloads = {}
    for i, f in enumerate(f_refs):
        w = {"t_sol_cycles": 100, "t_sol_ms": 0.001}
        if f is not None:
            w["f_ref_mhz"] = f
        workloads[f"uuid-{i}"] = w
    workloads["uuid-failed"] = {"error": "RuntimeError: traced nothing"}
    return {"status": "ok", "workloads": workloads}


def test_header_states_the_arch_clock_when_every_record_agrees():
    results = {"L1__001": _problem(2400.0, 2400.0), "L2__002": _problem(2400.0)}
    assert sol_bounds._header_f_ref(results, 2400.0) == (2400.0, [2400.0], 0)


def test_header_refuses_to_pick_when_the_body_holds_two_clocks():
    """The mixed file: 96 records at 2.4 GHz among 2902 at 1.8, header 2400."""
    results = {"L1__005": _problem(2400.0), "L1__037": _problem(1800.0, 1800.0)}
    header, observed, unstamped = sol_bounds._header_f_ref(results, 2400.0)
    assert header is None
    assert observed == [1800.0, 2400.0] and unstamped == 0


def test_header_refuses_a_body_that_is_uniform_but_not_at_the_arch_clock():
    """D63 exactly: one clock throughout the body, another one in the header.

    The file was not internally inconsistent when it first shipped -- it was
    uniformly at 1.8 GHz under a header of 2400, because the header came from
    `--freq-mhz` and the body came from a resumed cache. Deriving the header
    from the arch config the run actually used is what makes this detectable.
    """
    results = {"L1__037": _problem(1800.0, 1800.0, 1800.0)}
    assert sol_bounds._header_f_ref(results, 2400.0) == (None, [1800.0], 0)


def test_header_refuses_a_body_with_unstamped_records():
    """Silence is not agreement, however plausible the header's number is."""
    results = {"L1__037": _problem(2400.0, None, None)}
    assert sol_bounds._header_f_ref(results, 2400.0) == (None, [2400.0], 2)


def test_a_file_with_no_bounds_at_all_may_still_state_its_derivation_clock():
    """Nothing to contradict the arch config, and the gap is reported elsewhere."""
    results = {"L1__037": {"status": "failed", "workloads": {}}}
    assert sol_bounds._header_f_ref(results, 2400.0) == (2400.0, [], 0)


def test_resume_reuses_a_cache_only_at_the_clock_it_was_computed_at():
    """The root cause: the per-problem cache is keyed by problem, not by clock.

    A resume that serves a 1.8 GHz body to a 2.4 GHz run is how a header came
    to describe an invocation while the body described a cache. Both the
    wrong-clock and the does-not-say cases must miss.
    """
    assert sol_bounds._resume_clock_mismatch(_problem(2400.0, 2400.0), 2400.0) is None
    assert "1800" in sol_bounds._resume_clock_mismatch(_problem(1800.0), 2400.0)
    assert "before" in sol_bounds._resume_clock_mismatch(_problem(None), 2400.0)
    # A cached failure has no bounds to be at the wrong clock, and re-running it
    # is already what resume does -- so this must not be reported as a clock
    # mismatch on top.
    assert sol_bounds._resume_clock_mismatch(
        {"status": "failed", "workloads": {}}, 2400.0) is None
