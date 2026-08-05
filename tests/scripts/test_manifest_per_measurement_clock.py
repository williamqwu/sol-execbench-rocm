# SPDX-License-Identifier: Apache-2.0
"""The manifest must anchor each bound at the clock its T_b was measured at.

Unlocked, frequency is a property of the kernel rather than of the node, so a
workload's T_b and the bound it is scored against have to be expressed at the same
frequency. Two things can go wrong quietly:

* a bound left at the reference clock while T_b came from a different one, which
  rescales that workload's score by the ratio; and
* the two bound candidates compared against each other in cycles while sitting at
  different frequencies, where a cycle count is not a common unit.

The regression guard matters as much as the new behaviour: rebuilding the published
MI355X-v1 manifest with these changes reproduces it exactly apart from one added
counter, so the fixed basis is provably untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_manifest as bm  # noqa: E402

DRAM_BPS = 8.0e12


def _solar_rec(compute_cycles: float, memory_bytes: int, f_ref: float = 1650.0):
    """A T_SOL record as sol_bounds.py now emits it."""
    mem_cycles = memory_bytes * f_ref * 1e6 / DRAM_BPS
    exact = max(compute_cycles, mem_cycles)
    import math
    return {
        "t_sol_cycles": max(1, math.ceil(exact)),
        "t_sol_cycles_exact": exact,
        "t_sol_ms": max(1, math.ceil(exact)) / (f_ref * 1e3),
        "compute_cycles": compute_cycles,
        "memory_bytes": memory_bytes,
        "dram_byte_per_sec": DRAM_BPS,
    }


# --------------------------------------------------------------- _at_clock


def test_no_clock_leaves_the_record_alone():
    """The fixed basis passes no frequency, and must get its record back untouched."""
    rec = _solar_rec(1e6, 1000)
    assert bm._at_clock(rec, None) is rec


def test_compute_bound_rescales_by_the_clock_ratio():
    rec = _solar_rec(compute_cycles=1e6, memory_bytes=1_000)   # compute dominates
    out = bm._at_clock(rec, 1743.0)
    assert out["t_sol_rescaled"] is True
    assert out["bottleneck_at_measured_clock"] == "compute"
    assert out["t_sol_ms"] == pytest.approx(rec["t_sol_ms"] * 1650 / 1743, rel=1e-3)


def test_traffic_bound_keeps_its_time_and_only_moves_cycles():
    """bytes/bandwidth does not run off the core clock, so its TIME must not move."""
    rec = {"t_sol_ms": 0.5, "t_sol_cycles": 825_000}
    out = bm._at_clock(rec, 1743.0, f_independent_time=True)
    assert out["t_sol_ms"] == 0.5
    assert out["t_sol_cycles"] == pytest.approx(0.5 * 1743 * 1e3, abs=1)


def test_unrescalable_record_is_marked_not_guessed():
    """A pre-split T_SOL carries only the max of the two terms. Inferring from
    `bottleneck` would hold only while the measured clock stayed above the reference
    one, and would be wrong the first time it did not."""
    out = bm._at_clock({"t_sol_ms": 0.5, "t_sol_cycles": 825}, 1743.0)
    assert out["t_sol_rescaled"] is False
    assert "sol_bounds" in out["t_sol_rescale_error"]
    assert out["t_sol_ms"] == 0.5          # left as-is rather than silently moved


# --------------------------------------------------------------- combine_bounds


def test_bound_is_placed_at_the_measured_clock():
    sol = {"P": {"w1": _solar_rec(1e6, 1_000)}}
    tb = {"P": {"w1": {"variant": "v1", "t_b_ms": 5.0, "f_for_bound_mhz": 1743.0}}}
    out, stats = bm.combine_bounds(sol, {}, tb)
    assert stats["rescaled_to_measured_clock"] == 1
    assert out["P"]["w1"]["f_for_bound_mhz"] == 1743.0
    assert out["P"]["w1"]["t_sol_ms"] == pytest.approx(1e6 / 1743 / 1e3, rel=1e-3)


def test_candidates_are_compared_in_time_not_cycles():
    """The traffic candidate here is the larger bound in TIME but, expressed at the
    reference clock while the solar one sits at the measured clock, the smaller in
    cycles. Comparing cycles would pick the weaker bound."""
    solar = {"P": {"w1": _solar_rec(compute_cycles=1e6, memory_bytes=1_000)}}
    traffic = {"P": {"w1": {"t_sol_cycles": 990_000, "t_sol_ms": 0.6}}}
    tb = {"P": {"w1": {"variant": "v1", "t_b_ms": 5.0, "f_for_bound_mhz": 1743.0}}}
    out, stats = bm.combine_bounds(solar, traffic, tb)
    # solar at 1743 is 1e6/1743/1e3 = 0.574 ms; traffic is 0.6 ms and so binds.
    assert out["P"]["w1"]["t_sol_ms"] == pytest.approx(0.6, rel=1e-6)
    assert stats["max_of_both"] == 1


def test_a_bound_above_t_b_is_still_rejected_after_rescaling():
    """Rescaling must not smuggle a bound past the sanity check that keeps scores
    below 1: the comparison against T_b has to happen in the rescaled frame."""
    solar = {"P": {"w1": _solar_rec(compute_cycles=1e7, memory_bytes=1_000)}}
    tb = {"P": {"w1": {"variant": "v1", "t_b_ms": 0.001, "f_for_bound_mhz": 1743.0}}}
    out, stats = bm.combine_bounds(solar, {}, tb)
    assert stats["solar_rejected_above_t_b"] == 1
    assert "P" not in out


# --------------------------------------------------------------- collect_t_b


def _write_tb(d: Path, name: str, prov: dict, winners: dict) -> None:
    (d / f"{name}.json").write_text(json.dumps(
        {"_provenance": prov, "problem": name, "winner_by_workload": winners}))


def test_unlocked_artifacts_are_accepted_despite_a_mismatched_f_lock(tmp_path):
    """F_LOCK is only a table entry for an unlocked artifact. Comparing against it,
    as the fixed basis does, would reject every one of them."""
    _write_tb(tmp_path, "P", {"clock_basis": "unlocked", "f_lock_mhz": 1650},
              {"w1": {"variant": "v1", "t_b_ms": 5.0,
                      "f_for_bound_mhz": 1743.0, "clock_stable": True}})
    got = bm.collect_t_b(tmp_path, 1650)
    assert got["P"]["w1"]["f_for_bound_mhz"] == 1743.0


def test_unlocked_winner_without_a_clock_is_dropped(tmp_path):
    """Its timing is real, but no frequency describes it, so no bound can be placed.
    Dropping beats anchoring it at the table value."""
    _write_tb(tmp_path, "P", {"clock_basis": "unlocked", "f_lock_mhz": 1650},
              {"w1": {"variant": "v1", "t_b_ms": 5.0, "f_for_bound_mhz": None,
                      "clock_stable": False},
               "w2": {"variant": "v1", "t_b_ms": 6.0, "f_for_bound_mhz": 1743.0,
                      "clock_stable": True}})
    got = bm.collect_t_b(tmp_path, 1650)
    assert set(got["P"]) == {"w2"}


def test_fixed_basis_still_rejects_a_foreign_clock(tmp_path):
    """The guard that caught 87 MI350X files merged into an MI355X directory."""
    _write_tb(tmp_path, "P", {"f_lock_mhz": 1300}, {"w1": {"t_b_ms": 5.0}})
    _write_tb(tmp_path, "Q", {"f_lock_mhz": 1650}, {"w1": {"t_b_ms": 5.0}})
    got = bm.collect_t_b(tmp_path, 1650)
    assert set(got) == {"Q"}
