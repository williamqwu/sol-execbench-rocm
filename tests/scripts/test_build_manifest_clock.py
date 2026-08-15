# SPDX-License-Identifier: Apache-2.0
"""The manifest must carry the clock, and must refuse to invent one.

Two things are being pinned here, and they pull in opposite directions, which is
why they are tested together:

* **The unlocked basis has to build at all.** `provenance.f_lock_mhz()` resolves
  to None on MI355X by design (docs/TODO-MI355X.md §3.3), and `build_manifest.py`
  hard-exits on that. Under `SOLEXBENCH_CLOCK_BASIS=unlocked` it must instead
  build from the per-measurement bracketed clocks.
* **It must still refuse.** The unlocked basis is not the permissive option: an
  artifact carrying no clock evidence must not become a scoreable anchor. An
  unknown clock is not a permissive one — the same reading the F_LOCK guard
  applies, moved down to where the clock varies.

  What that no longer covers is a bracket refused for *spread*. Under the interval
  methodology those two samples are evidence, not an absence, and the measurement
  is admitted with a published width. The refusal is kept as a label and still
  counted; only its consequence changed.

Plus the two plumbing properties nothing else would catch: the four re-clocking
terms reaching the per-workload record (§4.2(c)), and the max-of-both tier
carrying terms that reproduce BOTH tiers at another clock (§4.2(b)).

CPU-only: everything is built from JSON on disk.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_manifest as bm  # noqa: E402
from solexbench_rocm.t_sol_at import t_sol_cycles_at  # noqa: E402

DRAM_BPS = 8.0e12
F_REF = 1650.0
UUID = "wl-0"
KEY = "L1__demo"


def _bracket(before=1800.0, after=1806.0, refused=False, reason=None,
             clock=None, threshold=0.0078):
    return {
        "clock_before_mhz": before,
        "clock_after_mhz": after,
        "clock_mhz": clock if clock is not None else (
            None if before is None or after is None else (before + after) / 2),
        "clock_bracket_spread": (
            None if before is None or after is None
            else abs(after - before) / ((after + before) / 2)),
        "clock_bracket_threshold": threshold,
        "clock_bracket_refused": refused,
        "clock_bracket_refused_reason": reason,
        "window_ns": [1_000, 13_000_000],
        "window_ms": 12.999,
    }


def _tb_artifact(tmp: Path, winners: dict, f_lock=None, name="p.json") -> Path:
    d = tmp / "tb"
    d.mkdir(exist_ok=True)
    (d / name).write_text(json.dumps({
        "_provenance": {"f_lock_mhz": f_lock, "utc": "now", "git_sha": "x"},
        "problem": KEY,
        "winner_by_workload": winners,
    }))
    return d


def _sol_doc(**overrides) -> dict:
    w = {"t_sol_cycles": 1000, "t_sol_ms": 1000 / (F_REF * 1e3),
         "bottleneck": "compute", "compute_cycles": 1000.0,
         "memory_bytes": 4096, "dram_byte_per_sec": DRAM_BPS,
         "mac_per_cycle": 4096.0}
    w.update(overrides)
    return {"problems": {KEY: {"workloads": {UUID: w}}}}


# ------------------------------------------------------ collect_t_b, unlocked


def test_unlocked_admits_a_bracketed_anchor(tmp_path):
    d = _tb_artifact(tmp_path, {UUID: {"variant": "v1_eager", "t_b_ms": 1.0,
                                       **_bracket()}})
    got = bm.collect_t_b(d, None, "unlocked")
    assert got[KEY][UUID]["clock_mhz"] == 1803.0


@pytest.mark.parametrize("winner,why", [
    ({"variant": "v1", "t_b_ms": 1.0}, "no bracket at all"),
    ({"variant": "v1", "t_b_ms": 1.0,
      **_bracket(before=None, after=None, refused=True,
                 reason="no_clock_evidence")},
     "clock unreadable"),
])
def test_unlocked_refuses_an_anchor_without_usable_clock_evidence(
        tmp_path, winner, why):
    """The failure this whole mechanism exists to prevent: a T_b that sets the
    score scale at a clock nobody can name.

    Note what is deliberately NOT in this list any more: a bracket refused for
    *spread*. That window has two measured clocks, and under the interval
    methodology it anchors with a stated width instead of being discarded. A
    window with no samples still cannot, because there is no width to state.
    """
    d = _tb_artifact(tmp_path, {UUID: winner})
    assert bm.collect_t_b(d, None, "unlocked") == {}, why


def test_unlocked_admits_a_spread_refused_anchor_and_keeps_the_label(tmp_path):
    """Refusal demoted from gate to label: the measurement survives, the flag does.

    The old rule dropped this anchor entirely, which is how five problems on the
    MI355X corpus ended up with no T_b at all while their timings sat in the
    artifact.
    """
    d = _tb_artifact(tmp_path, {UUID: {
        "variant": "v1", "t_b_ms": 1.0,
        **_bracket(before=1607.0, after=2148.0, refused=True,
                   reason="bracket_spread_above_threshold")}})
    got = bm.collect_t_b(d, None, "unlocked")
    assert got[KEY][UUID]["t_b_ms"] == 1.0
    assert got[KEY][UUID]["clock_bracket_refused"] is True
    assert (got[KEY][UUID]["clock_bracket_refused_reason"]
            == "bracket_spread_above_threshold")


def test_unlocked_keeps_the_good_measurements_out_of_a_mixed_artifact(tmp_path):
    """A workload with no clock at all must not take its siblings down with it —
    that would turn one unreadable sample into a lost problem."""
    d = _tb_artifact(tmp_path, {
        "good": {"variant": "v1", "t_b_ms": 1.0, **_bracket()},
        "bad": {"variant": "v1", "t_b_ms": 2.0,
                **_bracket(before=None, after=None, refused=True,
                           reason="no_clock_evidence")},
    })
    got = bm.collect_t_b(d, None, "unlocked")
    assert set(got[KEY]) == {"good"}


def test_locked_basis_is_unchanged_by_any_of_this(tmp_path):
    """The MI350X corpus must build exactly as it did. Its artifacts carry no
    brackets at all, and under the locked basis that is not a defect."""
    d = _tb_artifact(tmp_path, {UUID: {"variant": "v1", "t_b_ms": 1.0}},
                     f_lock=1300)
    assert bm.collect_t_b(d, 1300, "locked")[KEY][UUID]["t_b_ms"] == 1.0
    # ...and the foreign-clock guard still rejects, which the unlocked path
    # replaces rather than relaxes.
    assert bm.collect_t_b(d, 1640, "locked") == {}


def test_a_null_f_lock_stamp_is_still_admitted_under_the_locked_basis(tmp_path):
    """build_manifest.py:162's deliberate admission. A null stamp means the
    artifact predates F_LOCK stamping, which is a different problem from being
    measured at the wrong clock — and it is the opposite reading to the
    top-level guard, which refuses on None. Both are intentional."""
    d = _tb_artifact(tmp_path, {UUID: {"variant": "v1", "t_b_ms": 1.0}},
                     f_lock=None)
    assert bm.collect_t_b(d, 1300, "locked")[KEY][UUID]["t_b_ms"] == 1.0


# ------------------------------------------------- the re-clocking terms


def test_the_four_reclocking_terms_reach_the_manifest_record():
    """§4.2(c). Until they do, `t_sol_at` can never see them however correct it
    is, and every record raises MissingBoundTerms."""
    merged, _ = bm.combine_bounds(_sol_doc()["problems"][KEY]["workloads"] and
                                  {KEY: _sol_doc()["problems"][KEY]["workloads"]},
                                  {}, {})
    rec = merged[KEY][UUID]
    for f in bm.RECLOCK_TERM_FIELDS:
        assert rec.get(f) is not None, f
    assert t_sol_cycles_at(rec, F_REF) == 1000
    # The clock provenance travels in the same list but is allowed to be null:
    # a tier that never stated its reference clock is a fact about the tier, and
    # `t_sol_at.bound_ms` is the thing that refuses on it. What must not happen is
    # the field going missing, because then nothing can refuse.
    for f in bm.CLOCK_PROVENANCE_FIELDS:
        assert f in rec, f


def test_max_of_both_reproduces_both_tiers_at_another_clock():
    """§4.2(b), the real gap.

    The declared-traffic tier carries no compute term. A `max_of_both` record
    that inherited only the winning tier's terms would re-clock as if the
    workload had no arithmetic — silently, and only at clocks other than the
    reference one, which is exactly where nobody would look.
    """
    solar = {KEY: {UUID: {"t_sol_cycles": 1000, "t_sol_ms": 1e-3,
                          "compute_cycles": 1000.0, "memory_bytes": 4096,
                          "dram_byte_per_sec": DRAM_BPS,
                          "mac_per_cycle": 4096.0}}}
    # A traffic bound that WINS at the reference clock (more bytes), and which
    # by construction has no compute term.
    traffic = {KEY: {UUID: {"t_sol_cycles": 2000, "t_sol_ms": 2e-3,
                            "compute_cycles": 0.0, "memory_bytes": 8192,
                            "dram_byte_per_sec": DRAM_BPS,
                            "mac_per_cycle": None}}}
    merged, stats = bm.combine_bounds(solar, traffic, {})
    rec = merged[KEY][UUID]
    assert rec["t_sol_source"] == "max_of_both"
    assert stats["reclock_terms_unioned"] == 1
    assert rec["compute_cycles"] == 1000.0, "SOLAR's arithmetic must survive"
    assert rec["memory_bytes"] == 8192, "the larger traffic term must survive"

    def two_tier_max(f_mhz):
        return max(t_sol_cycles_at(solar[KEY][UUID], f_mhz),
                   t_sol_cycles_at(traffic[KEY][UUID], f_mhz))

    # Across a range that spans this part's own unlocked clocks (1800 MHz on a
    # dense GEMM, 2392 MHz on a small one) and well past them in both directions.
    for f in (200.0, 800.0, 1650.0, 1800.0, 2392.0, 5000.0):
        assert t_sol_cycles_at(rec, f) == two_tier_max(f), f


def test_conflicting_bandwidths_leave_the_record_unreclockable():
    """Two bandwidths means two arch configs produced these tiers. Refusing to
    merge leaves `t_sol_at` raising, which is visible; picking one would produce
    a plausible bound at every clock but the reference one."""
    solar = {KEY: {UUID: {"t_sol_cycles": 1000, "t_sol_ms": 1e-3,
                          "compute_cycles": 1000.0, "memory_bytes": 4096,
                          "dram_byte_per_sec": DRAM_BPS}}}
    traffic = {KEY: {UUID: {"t_sol_cycles": 2000, "t_sol_ms": 2e-3,
                            "compute_cycles": 0.0, "memory_bytes": 8192,
                            "dram_byte_per_sec": DRAM_BPS * 1.5}}}
    merged, stats = bm.combine_bounds(solar, traffic, {})
    assert stats["reclock_terms_conflicting_bandwidth"] == 1
    assert "dram_byte_per_sec" not in merged[KEY][UUID]


def test_two_printings_of_one_bandwidth_are_not_a_conflict():
    """The MI355X tiers really emit these two numbers, and they are one number.

    7999919999999.999 and 7999920000000.0 are 7.99992e12 reached by a division and
    by a multiplication. Exact set equality read them as two arch configs and left
    every two-tier MI355X record un-re-clockable -- 16 of 16 workloads on
    L2__004_fused_residual_rms_mlp, i.e. no interval on the widest problem in the
    corpus. The guard is for two configs; no two configs differ by 2e-16.
    """
    solar = {KEY: {UUID: {"t_sol_cycles": 1000, "t_sol_ms": 1e-3,
                          "compute_cycles": 1000.0, "memory_bytes": 4096,
                          "dram_byte_per_sec": 7999919999999.999}}}
    traffic = {KEY: {UUID: {"t_sol_cycles": 2000, "t_sol_ms": 2e-3,
                            "compute_cycles": 0.0, "memory_bytes": 8192,
                            "dram_byte_per_sec": 7999920000000.0}}}
    merged, stats = bm.combine_bounds(solar, traffic, {})
    assert not stats.get("reclock_terms_conflicting_bandwidth")
    rec = merged[KEY][UUID]
    assert rec["compute_cycles"] == 1000.0 and rec["memory_bytes"] == 8192
    assert t_sol_cycles_at(rec, F_REF) > 0
    # ...and a real disagreement is still a conflict, at any size worth calling one.
    traffic[KEY][UUID]["dram_byte_per_sec"] = 7999920000000.0 * 1.001
    _, stats2 = bm.combine_bounds(solar, traffic, {})
    assert stats2["reclock_terms_conflicting_bandwidth"] == 1


def test_a_record_predating_the_split_is_counted_not_faked():
    solar = {KEY: {UUID: {"t_sol_cycles": 1000, "t_sol_ms": 1e-3}}}
    merged, stats = bm.combine_bounds(solar, {}, {})
    assert stats["reclock_terms_missing"] == 1
    assert "compute_cycles" not in merged[KEY][UUID]


# ------------------------------------------------------ end to end, on disk


def _build(tmp_path, env_extra: dict, winners: dict, sol=None):
    tb = _tb_artifact(tmp_path, winners)
    (tmp_path / "t_sol.json").write_text(json.dumps(sol or _sol_doc()))
    (tmp_path / "t_sol_traffic.json").write_text(json.dumps({"problems": {}}))
    (tmp_path / "deferred.json").write_text(json.dumps({"problems": {}}))
    (tmp_path / "tol").mkdir(exist_ok=True)
    data = tmp_path / "data" / "L1" / "demo"
    data.mkdir(parents=True, exist_ok=True)
    (data / "definition.json").write_text("{}")
    out = tmp_path / "manifest.json"
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
           "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}", **env_extra}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_manifest.py"),
         "--out", str(out), "--t-sol", str(tmp_path / "t_sol.json"),
         "--t-sol-traffic", str(tmp_path / "t_sol_traffic.json"),
         "--t-b", str(tb), "--tolerances", str(tmp_path / "tol"),
         "--deferred", str(tmp_path / "deferred.json"),
         "--data", str(tmp_path / "data")],
        capture_output=True, text=True, env=env, timeout=300)
    return proc, out


def test_the_unlocked_basis_builds_where_the_locked_one_refuses(tmp_path):
    """The two halves of §3.4, in one test.

    Off-GPU with no override, `provenance.f_lock_mhz()` is None and the locked
    build must hard-exit before writing anything. The same inputs on the
    unlocked basis must produce a manifest — from the brackets, not from a
    default.
    """
    winners = {UUID: {"variant": "v1_eager", "t_b_ms": 1.0, **_bracket()}}

    locked, out = _build(tmp_path, {}, winners)
    assert locked.returncode != 0
    assert "cannot resolve F_LOCK" in locked.stderr
    assert not out.exists(), "a refused build must write nothing at all"

    unlocked, out = _build(tmp_path, {"SOLEXBENCH_CLOCK_BASIS": "unlocked"},
                           winners)
    assert unlocked.returncode == 0, unlocked.stderr
    doc = json.loads(out.read_text())
    assert doc["clock_basis"] == "unlocked"

    w = doc["problems"][KEY]["workloads"][UUID]
    assert w["scoreable"] is True
    # The bracket, as fields rather than as prose.
    assert w["clock_before_mhz"] == 1800.0 and w["clock_after_mhz"] == 1806.0
    assert w["clock_mhz"] == 1803.0
    assert w["clock_bracket_threshold"] == 0.0078
    assert w["clock_bracket_refused"] is False
    assert w["clock_bracket_spread"] == pytest.approx(6 / 1803)
    assert w["window_ns"] == [1_000, 13_000_000]
    # ...and the terms that make the bracket usable.
    for f in bm.RECLOCK_TERM_FIELDS:
        assert w[f] is not None, f
    for f in bm.CLOCK_PROVENANCE_FIELDS:
        assert f in w, f
    assert t_sol_cycles_at(w, w["clock_mhz"]) == 1000

    # The refusal rate is a first-class field on the manifest too.
    assert doc["clock_bracket"]["n_refused"] == 0
    assert doc["clock_bracket"]["refusal_rate"] == 0.0
    assert doc["clock_bracket"]["n_with_clock_evidence"] == 1


def test_the_unlocked_basis_refuses_a_corpus_with_no_clock_evidence(tmp_path):
    """The line that keeps "unlocked" from meaning "unchecked"."""
    proc, out = _build(tmp_path, {"SOLEXBENCH_CLOCK_BASIS": "unlocked"},
                       {UUID: {"variant": "v1_eager", "t_b_ms": 1.0}})
    assert proc.returncode != 0
    assert "no usable clock bracket" in proc.stderr
    assert not out.exists()


def test_an_unknown_basis_is_refused_rather_than_defaulted(tmp_path):
    """A typo in the basis would otherwise silently build the other one."""
    proc, _ = _build(tmp_path, {"SOLEXBENCH_CLOCK_BASIS": "unlokced"},
                     {UUID: {"variant": "v1", "t_b_ms": 1.0, **_bracket()}})
    assert proc.returncode != 0 and "not a basis" in proc.stderr


def test_a_measured_f_lock_still_builds_on_the_locked_basis(tmp_path):
    """The escape hatch §3.3 names must keep working, unchanged."""
    proc, out = _build(tmp_path, {"SOLEXBENCH_F_LOCK_MHZ": "1300"},
                       {UUID: {"variant": "v1", "t_b_ms": 1.0}})
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text())
    assert doc["clock_basis"] == "locked"
    w = doc["problems"][KEY]["workloads"][UUID]
    assert w["scoreable"] is True and w["clock_mhz"] is None


def test_the_tier_max_is_taken_in_time_not_in_cycles():
    """The two tiers count cycles at different reference clocks.

    SOLAR's `t_sol_cycles` is at f_ref 1.8 GHz (4444.4 DRAM bytes/cycle); the
    declared-traffic tier's is at the arch config's 2.4 GHz (3333.3). So a
    workload can have MORE traffic cycles and LESS time, and `t_cyc > s_cyc`
    then picks the smaller of two lower bounds — the max is not a max. On
    MI355X that happened on 255 of the 2796 workloads where both tiers
    survived, and it is the invisible direction: a T_SOL below the true bound
    inflates S for every submission slower than T_b.

    Same bytes, both tiers, is the sharpest case: 16384 B is 3.69 SOLAR cycles
    (-> 4) and 4.92 traffic cycles (-> 5), so cycles say traffic wins while
    time says SOLAR does, by the full 2.4/1.8.
    """
    solar = {KEY: {UUID: {"t_sol_cycles": 4, "t_sol_ms": 2.222e-6,
                          "compute_cycles": 0.0, "memory_bytes": 16384,
                          "dram_byte_per_sec": DRAM_BPS,
                          "mac_per_cycle": 4096.0}}}
    traffic = {KEY: {UUID: {"t_sol_cycles": 5, "t_sol_ms": 2.083e-6,
                            "compute_cycles": 0.0, "memory_bytes": 16384,
                            "dram_byte_per_sec": DRAM_BPS,
                            "mac_per_cycle": None}}}
    rec = bm.combine_bounds(solar, traffic, {})[0][KEY][UUID]
    assert rec["t_sol_source"] == "solar_fused"
    assert rec["t_sol_ms"] == 2.222e-6, "the larger bound in TIME must win"


def test_a_surviving_tier_without_a_time_is_refused_rather_than_compared():
    """Falling back to a cycle comparison would reintroduce the unit error on
    exactly the records that cannot be checked. Raise instead."""
    solar = {KEY: {UUID: {"t_sol_cycles": 4, "compute_cycles": 0.0,
                          "memory_bytes": 16384,
                          "dram_byte_per_sec": DRAM_BPS}}}
    traffic = {KEY: {UUID: {"t_sol_cycles": 5, "t_sol_ms": 2.083e-6,
                            "compute_cycles": 0.0, "memory_bytes": 16384,
                            "dram_byte_per_sec": DRAM_BPS}}}
    with pytest.raises(ValueError, match="no t_sol_ms"):
        bm.combine_bounds(solar, traffic, {})
