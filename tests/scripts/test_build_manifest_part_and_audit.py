# SPDX-License-Identifier: Apache-2.0
"""Two things the manifest could not say about itself, and now does.

**Which part it is for.** `manifest-v2` and `-v3` shipped with no `part` key at
all, and every consumer that needed one -- the scorer's part guard, the
leaderboard ingest, task 03's check D -- recovered it from the torch device names
in the provenance block. That is an inference from WHERE THE FILE WAS WRITTEN, in
a tree that holds two parts' artifacts side by side, and it is the third defect
listed under Issue 7. The inputs know the answer; the builder now asks them and
refuses when they disagree.

**Which tier was thrown out.** `combine_bounds` has always computed
`t_sol_tier_rejected_above_t_b` -- the tier whose candidate sat above the measured
T_b -- and a test has always asserted it, but the field stopped at the manifest's
record writer and reached 0 of 3957 published workloads. Without it the manifest
cannot distinguish "this bound is one tier's because the other was SMALLER" from
"this bound is one tier's because the other was IMPOSSIBLE", which are different
claims about how well determined the number is.

CPU-only. No GPU, no dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_manifest as bm  # noqa: E402

BPS = 7999920000000.0
KEY = "L2__002_decoder_layer_full_block"
UUID = "u0"


# -- the part declaration ---------------------------------------------------

def _write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(__import__("json").dumps(doc))
    return p


def test_a_top_level_part_is_taken_over_the_device_names(tmp_path):
    """`sol_bounds.py` writes `part` at the top level; that is a statement about
    the file, where the device list is evidence about the node it was made on."""
    a = _write(tmp_path, "a.json", {"part": "MI355X", "problems": {}})
    assert bm._inputs_part(a) == "MI355X"


def test_the_device_names_are_the_fallback_not_the_first_answer(tmp_path):
    """`t_sol_traffic.json` states nothing today, so the fallback has to work --
    but it is a fallback, and the test says which is which."""
    b = _write(tmp_path, "b.json", {
        "_provenance": {"torch": {"devices": ["AMD Instinct MI355X"] * 8}},
        "problems": {}})
    assert bm._inputs_part(b) == "MI355X"


def test_inputs_that_name_two_parts_are_a_build_error(tmp_path):
    """A manifest pairing one part's bounds with another part's anchors would be
    scored against silently. This is the failure the whole part-declaration work
    exists to make loud, so it must not resolve to a majority vote."""
    a = _write(tmp_path, "a.json", {"part": "MI355X", "problems": {}})
    b = _write(tmp_path, "b.json", {"part": "MI350X", "problems": {}})
    with pytest.raises(SystemExit) as e:
        bm._inputs_part(a, b)
    msg = str(e.value)
    assert "MI355X" in msg and "MI350X" in msg


def test_inputs_that_say_nothing_leave_the_part_unstated(tmp_path):
    """None is a legible absence, not a guess. The MI350X release inputs predate
    device names in provenance and would otherwise have to be invented for."""
    a = _write(tmp_path, "a.json", {"problems": {}})
    assert bm._inputs_part(a) is None


def test_a_missing_input_file_is_not_evidence_of_anything(tmp_path):
    assert bm._inputs_part(tmp_path / "absent.json") is None


# -- the rejected-tier field ------------------------------------------------

def _tiers(solar_cycles, traffic_cycles, f_ref=2400.0):
    def rec(cycles, compute, membytes):
        return {"t_sol_cycles": cycles, "t_sol_ms": cycles / (f_ref * 1e3),
                "compute_cycles": compute, "memory_bytes": membytes,
                "dram_byte_per_sec": BPS, "f_ref_mhz": f_ref,
                "bottleneck": "compute" if compute >= membytes else "memory"}
    solar = {KEY: {UUID: rec(solar_cycles, float(solar_cycles), 4096)}}
    traffic = {KEY: {UUID: rec(traffic_cycles, 0.0, traffic_cycles * 3333.3)}}
    return solar, traffic


def test_the_rejected_tier_is_recorded_when_one_is_thrown_out():
    """SOLAR at 24,000,000 cycles is 10 ms at 2.4 GHz, well above a 1 ms T_b, so
    it is not a lower bound at all and the gate drops it."""
    solar, traffic = _tiers(24_000_000, 1_200_000)
    tb = {KEY: {UUID: {"t_b_ms": 1.0, "clock_before_mhz": 2400.0,
                       "clock_after_mhz": 2400.0}}}
    out, stats = bm.combine_bounds(solar, traffic, tb)
    rec = out[KEY][UUID]
    assert rec["t_sol_tier_rejected_above_t_b"] == ["solar_fused"]
    assert stats["solar_rejected_above_t_b"] == 1


def test_no_rejection_leaves_the_field_null_rather_than_an_empty_list():
    """Null, not `[]`: a consumer filtering on truthiness gets the same answer
    either way, but a reader should not have to know that."""
    solar, traffic = _tiers(1_200_000, 1_000_000)
    tb = {KEY: {UUID: {"t_b_ms": 1.0, "clock_before_mhz": 2400.0,
                       "clock_after_mhz": 2400.0}}}
    out, _ = bm.combine_bounds(solar, traffic, tb)
    assert out[KEY][UUID]["t_sol_tier_rejected_above_t_b"] is None
