# SPDX-License-Identifier: Apache-2.0
"""`_tier_times_ms`'s counters must name the branch they count.

The D63 correction puts both T_SOL tiers on the measurement's own clock before
comparing them. Not every record can take that path: one whose SOLAR terms predate
the split into separately-scalable terms has nothing to re-evaluate, and falls back
to the stored millisecond columns — the mixed-clock columns D63 exists to retire.

Which of the two happened is the only statistic a reader has for sizing the
remaining exposure, and the first version of this code got it exactly backwards.
`stats["tier_compared_at_reference_clock"]` was incremented inside the `except`
branch, so `artifacts/09-MI355X/manifest-v4.json` published

    "tier_compared_at_reference_clock": 348

when 3369 records were compared at the reference clock and those 348 were the ones
that were *not*. A reader auditing "did the fix reach my records?" would have read
8.8% coverage as 85%, and would have been pointed at the still-exposed population
under a name saying it was the fixed one.

That is not a cosmetic mislabel, so it gets a test rather than a comment: the
inversion is invisible in every bound (no published number depends on the counter),
which is exactly the class of defect this repo keeps shipping.

CPU-only: hand-built records, no dataset, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_manifest as bm  # noqa: E402

DRAM_BPS = 8.0e12


def _terms(compute_cycles: float, memory_bytes: int) -> dict:
    """A T_SOL record carrying the terms needed to re-evaluate it at any clock."""
    return {
        "compute_cycles": compute_cycles,
        "memory_bytes": memory_bytes,
        "dram_byte_per_sec": DRAM_BPS,
        "t_sol_cycles": max(1, int(compute_cycles)),
        "t_sol_ms": compute_cycles / (1800.0 * 1e3),
    }


def _legacy() -> dict:
    """A record from before the term split: only the collapsed max survived."""
    return {"t_sol_cycles": 4096, "t_sol_ms": 4096 / (1800.0 * 1e3)}


def test_a_record_put_on_one_clock_counts_as_compared_not_as_fallback():
    stats: dict = {}
    s_ms, t_ms, at = bm._tier_times_ms(_terms(1000.0, 4096), _terms(10.0, 8192),
                                       2400.0, stats)

    assert at == 2400.0, "both tiers were re-evaluated, so the clock must be reported"
    assert s_ms is not None and t_ms is not None
    assert stats.get("tier_compared_at_reference_clock") == 1
    assert stats.get("tier_fell_back_to_stored_clock", 0) == 0


def test_a_record_that_cannot_be_re_evaluated_counts_as_a_fallback():
    """The inversion this file exists for: the except branch must NOT be counted
    as a comparison."""
    stats: dict = {}
    s_ms, t_ms, at = bm._tier_times_ms(_legacy(), _terms(10.0, 8192), 2400.0, stats)

    assert at is None, "the comparison was not made on one clock, and must say so"
    assert stats.get("tier_fell_back_to_stored_clock") == 1
    assert stats.get("tier_compared_at_reference_clock", 0) == 0, (
        "a record that fell back to its stored mixed-clock column was counted as "
        "one that had been put on the measurement's clock — the D63 defect, "
        "reported as its own fix"
    )
    # And it really did fall back to the stored columns, not to something new.
    assert s_ms == _legacy()["t_sol_ms"]


def test_no_measurement_clock_is_its_own_outcome():
    """Absent T_b clock is not a defective T_SOL record and must not be pooled
    with one."""
    stats: dict = {}
    _, _, at = bm._tier_times_ms(_terms(1000.0, 4096), _terms(10.0, 8192), None, stats)

    assert at is None
    assert stats.get("tier_no_measurement_clock") == 1
    assert stats.get("tier_fell_back_to_stored_clock", 0) == 0
    assert stats.get("tier_compared_at_reference_clock", 0) == 0


def test_the_three_outcomes_partition_every_call():
    """Whatever the mix, the counters must sum to the number of calls — otherwise
    a reader cannot tell coverage from silence."""
    stats: dict = {}
    records = [
        (_terms(1000.0, 4096), _terms(10.0, 8192), 2400.0),
        (_terms(500.0, 1024), _terms(10.0, 8192), 2385.0),
        (_legacy(), _terms(10.0, 8192), 2400.0),
        (_terms(1000.0, 4096), _terms(10.0, 8192), None),
    ]
    for s, t, f in records:
        bm._tier_times_ms(s, t, f, stats)

    counted = (stats.get("tier_compared_at_reference_clock", 0)
               + stats.get("tier_fell_back_to_stored_clock", 0)
               + stats.get("tier_no_measurement_clock", 0))
    assert counted == len(records)
    assert stats["tier_compared_at_reference_clock"] == 2
    assert stats["tier_fell_back_to_stored_clock"] == 1
    assert stats["tier_no_measurement_clock"] == 1


def test_a_one_tier_comparison_is_counted_as_one_tier():
    """"Compared at the reference clock" against nothing is still counted apart.

    A record with NO SOLAR entry cannot raise `MissingBoundTerms` -- there is
    nothing to raise about -- so it takes the success branch and joins
    `tier_compared_at_reference_clock`, while a record whose SOLAR entry is an
    *error* raises and joins the fallback. Both are SOLAR failures; without a
    sub-counter the split between the two headline numbers tracks which flavour
    occurred rather than what either name claims.

    On MI355X manifest-v4 this is 0 of 3360 (instrumented build), so the main
    counter is honest there today. This test is what keeps that a fact rather
    than an assumption.
    """
    stats: dict = {}
    _s, t_ms, at = bm._tier_times_ms(None, _terms(10.0, 8192), 2400.0, stats)
    assert at == 2400.0 and t_ms is not None
    assert stats["tier_compared_at_reference_clock"] == 1
    assert stats["tier_compared_one_tier_only"] == 1

    bm._tier_times_ms(_terms(10.0, 8192), None, 2400.0, stats)
    assert stats["tier_compared_one_tier_only"] == 2


def test_two_tier_comparisons_are_not_counted_as_one_tier():
    """The sub-counter must be silent on the ordinary case, or it says nothing.

    A counter that fires on every record would report "all of it is one-sided",
    which is the same failure mode as the inverted counter this file exists for:
    a number that reads as coverage and is not.
    """
    stats: dict = {}
    bm._tier_times_ms(_terms(1000.0, 4096), _terms(10.0, 8192), 2400.0, stats)
    assert stats["tier_compared_at_reference_clock"] == 1
    assert stats.get("tier_compared_one_tier_only", 0) == 0

    # A fallback is not a one-tier comparison either: no comparison was made.
    bm._tier_times_ms(_legacy(), _terms(10.0, 8192), 2400.0, stats)
    assert stats["tier_fell_back_to_stored_clock"] == 1
    assert stats.get("tier_compared_one_tier_only", 0) == 0
