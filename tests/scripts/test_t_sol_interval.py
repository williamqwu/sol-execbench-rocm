# SPDX-License-Identifier: Apache-2.0
"""The interval methodology, where it reaches the manifest and the score.

`tests/scripts/test_t_sol_at.py` pins the arithmetic. This pins the consequences,
which are the part that can silently regress:

* **S inherits the interval.** The published S is the one at the published bound
  (minimum clock), and the two ends bracket it. A score published without its width
  is a score that claims a precision the measurement does not have.
* **Refusal is a label, not a gate.** A bracket refused for spread keeps its flag,
  keeps being counted, and no longer discards the measurement. A bracket with no
  samples at all is still refused, and `clock_fatalities` still fails a run closed
  on the conditions it always did.
* **The locked path is byte-identical.** There is a frozen MI350X corpus of 3717
  workloads behind that sentence.

CPU-only: everything is JSON on disk and pure arithmetic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "runners"))
sys.path.insert(0, str(ROOT / "src"))

import build_manifest as bm  # noqa: E402
import score_solutions as ss  # noqa: E402
from sol_execbench.core.bench.clock_bracket import (  # noqa: E402
    clock_interval,
    has_clock_evidence,
    has_clock_interval,
    make_bracket,
    summarize_brackets,
)

DRAM_BPS = 8.0e12
F_REF = 1650.0
UUID = "wl-0"
KEY = "L1__demo"

#: L2__004_fused_residual_rms_mlp's measured bracket, and L1__013's, from
#: artifacts/06-MI355X. The contrast between them is the deliverable.
WIDE = (1607.0, 2148.0)
CLEAN = (2314.0, 2390.0)


def _bound(compute_cycles=1e5, memory_bytes=4096) -> dict:
    return {"compute_cycles": compute_cycles, "memory_bytes": memory_bytes,
            "dram_byte_per_sec": DRAM_BPS, "mac_per_cycle": 4096.0,
            "t_sol_ms": 1e-3, "t_sol_cycles": int(compute_cycles)}


def _bracket(before, after, *, refused=False, reason=None) -> dict:
    return {
        "clock_before_mhz": before,
        "clock_after_mhz": after,
        "clock_mhz": None if before is None else (before + after) / 2,
        "clock_bracket_spread": None if before is None else
        abs(after - before) / ((after + before) / 2),
        "clock_bracket_threshold": 0.0078,
        "clock_bracket_refused": refused,
        "clock_bracket_refused_reason": reason,
    }


def _wide_bracket() -> dict:
    return _bracket(*WIDE, refused=True, reason="bracket_spread_above_threshold")


# --------------------------------------------------------------- S as an interval


@pytest.mark.parametrize("t_k,t_b,why", [
    (0.5, 2.0, "kernel faster than the baseline anchor"),
    (3.0, 2.0, "kernel slower than the baseline anchor"),
])
def test_s_at_the_two_ends_brackets_the_published_s(t_k, t_b, why):
    """Both parametrisations, because S is not monotone in T_SOL the same way on
    each side of T_b: dS/dT_SOL carries the sign of (T_k - T_b). A test on only the
    fast case would pass while the naming of the ends was inverted."""
    iv = ss._interval_score(_bound(), _wide_bracket(), t_k, t_b)
    lo = iv["sol_score_at_clock_min"]
    hi = iv["sol_score_at_clock_max"]
    published = ss.sol_score(t_k, iv["t_sol_ms_published"], t_b)
    assert None not in (lo, hi, published), why
    assert min(lo, hi) <= published <= max(lo, hi), why
    # The published S is one of the ends, not something between them: it is
    # computed at a clock the card was actually observed at.
    assert published == pytest.approx(lo)


def test_the_published_bound_is_the_one_the_score_uses():
    iv = ss._interval_score(_bound(), _wide_bracket(), 0.5, 2.0)
    assert iv["t_sol_ms_published"] == iv["t_sol_ms_at_clock_min"]
    assert iv["t_sol_ms_published"] > iv["t_sol_ms_at_clock_max"]


def test_a_clean_bracket_gives_a_narrow_s_interval_and_a_wide_one_does_not():
    """The contrast this whole change exists to make visible."""
    wide = ss._interval_score(_bound(), _wide_bracket(), 0.5, 2.0)
    clean = ss._interval_score(_bound(), _bracket(*CLEAN), 0.5, 2.0)
    assert wide["t_sol_interval_halfwidth_rel"] > 0.1
    assert clean["t_sol_interval_halfwidth_rel"] < 0.02
    spread = lambda iv: abs(iv["sol_score_at_clock_min"]
                            - iv["sol_score_at_clock_max"])
    assert spread(wide) > 5 * spread(clean)


def test_no_interval_is_stated_rather_than_approximated():
    """A pre-split bound, or a window with no samples, yields nothing at all.

    `{}` and not a half-filled record: the caller then keeps the manifest's own
    reference-clock bound, and the absence is on the record. Inventing an endpoint
    is prime directive 1.
    """
    assert ss._interval_score(_bound(), _bracket(None, None), 0.5, 2.0) == {}
    assert ss._interval_score({"t_sol_ms": 1e-3}, _wide_bracket(), 0.5, 2.0) == {}


# ------------------------------------------- refusal: demoted, but never dropped


def test_a_wide_bracket_is_usable_and_an_absent_one_is_not():
    wide = make_bracket(*WIDE).as_dict()
    assert wide["clock_bracket_refused"] is True, "still refused, still labelled"
    assert has_clock_evidence(wide) is False, "still not a single-clock window"
    assert has_clock_interval(wide) is True, "but the two samples are evidence"
    assert clock_interval(wide) == WIDE

    for absent in (make_bracket(None, None).as_dict(),
                   make_bracket(1800.0, 1806.0, sampler_error="boom").as_dict(),
                   None, {}):
        assert has_clock_interval(absent) is False


def test_the_refusal_counts_are_unchanged_by_the_demotion():
    """`clock_fatalities` reads these three and must keep behaving identically."""
    records = [make_bracket(*WIDE).as_dict(),          # refused on spread
               make_bracket(2314.0, 2320.0).as_dict(),  # clean
               make_bracket(None, None).as_dict()]      # no clock at all
    s = summarize_brackets(records)
    assert s["n_bracketed"] == 3
    assert s["n_refused"] == 2
    assert s["refusal_rate"] == pytest.approx(2 / 3)
    assert s["refused_by_reason"] == {"bracket_spread_above_threshold": 1,
                                      "no_clock_evidence": 1}
    # ...and the new split, which is added beside them rather than replacing them.
    assert s["n_with_interval"] == 2
    assert s["n_refused_with_interval"] == 1
    assert s["n_without_interval"] == 1


def test_clock_fatalities_still_fails_a_run_with_no_clock_at_all():
    """The gate that stays a gate. Imported from the runner, not reimplemented."""
    from time_tb_candidates import clock_fatalities

    absent = [make_bracket(None, None).as_dict() for _ in range(4)]
    fatal = clock_fatalities(summarize_brackets(absent), absent,
                             attempted=4, bracketing=True)
    assert fatal, "a run whose sampler produced nothing must not exit 0"

    err = [make_bracket(1800.0, 1806.0, sampler_error="TypeError: x").as_dict()]
    assert clock_fatalities(summarize_brackets(err), err, attempted=1,
                            bracketing=True), "a sampler defect is still fatal"


# ------------------------------------------------ the manifest, end to end on disk


def _build(tmp_path, env_extra: dict, tb_doc: dict, sol: dict):
    d = tmp_path / "tb"
    d.mkdir(exist_ok=True)
    (d / "p.json").write_text(json.dumps(tb_doc))
    (tmp_path / "t_sol.json").write_text(json.dumps(sol))
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
         "--t-b", str(d), "--tolerances", str(tmp_path / "tol"),
         "--deferred", str(tmp_path / "deferred.json"),
         "--data", str(tmp_path / "data")],
        capture_output=True, text=True, env=env, timeout=300)
    return proc, out


def _tb_doc(winners: dict, variants: dict | None = None) -> dict:
    return {"_provenance": {"f_lock_mhz": None, "utc": "now", "git_sha": "x"},
            "problem": KEY, "winner_by_workload": winners,
            "variants": variants or {}}


def _sol_doc(bound: dict | None = None) -> dict:
    return {"problems": {KEY: {"workloads": {UUID: bound or _bound()}}}}


def test_the_manifest_carries_both_ends_the_width_and_the_bottleneck(tmp_path):
    proc, out = _build(
        tmp_path, {"SOLEXBENCH_CLOCK_BASIS": "unlocked"},
        _tb_doc({UUID: {"variant": "v1", "t_b_ms": 5.0, **_wide_bracket()}}),
        _sol_doc())
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text())
    w = doc["problems"][KEY]["workloads"][UUID]

    assert w["t_sol_clock_min_mhz"] == 1607.0
    assert w["t_sol_clock_max_mhz"] == 2148.0
    assert w["t_sol_ms_at_clock_min"] > w["t_sol_ms_at_clock_max"]
    assert w["t_sol_ms_published"] == w["t_sol_ms_at_clock_min"]
    assert w["t_sol_published_end"] == "clock_min"
    assert w["t_sol_interval_halfwidth_rel"] == pytest.approx(0.144, abs=5e-3)
    assert w["t_sol_bottleneck_at_clock_min"] == "compute"
    assert w["t_sol_bottleneck_flips"] is False
    # The label survives into the manifest beside the interval it produced.
    assert w["clock_bracket_refused"] is True

    # Sortable per problem and per corpus, without reopening any workload.
    p = doc["problems"][KEY]
    assert p["t_sol_interval_halfwidth_max"] == pytest.approx(0.144, abs=5e-3)
    assert p["n_workloads_with_t_sol_interval"] == 1
    assert doc["t_sol_interval"]["published_at"] == "clock_min"
    assert doc["t_sol_interval"]["problems_wide"] == [KEY]


def test_the_manifest_recovers_an_anchor_the_sweep_time_gate_dropped(tmp_path):
    """The five unanchorable problems. Every number comes from the artifact.

    The sweep refused all of this problem's brackets and wrote no winner, so under
    the old rule it reached the manifest as "missing T_b" while its timings sat in
    the `variants` block. The recovery reads them back and applies the runner's own
    rule -- fastest passing variant wins.
    """
    variants = {
        "v1_eager": {"ok": True, "all_passed": True,
                     "latency_ms_by_workload": {UUID: 9.0},
                     "clock_bracket_by_workload": {UUID: _wide_bracket()}},
        "v4_contiguous": {"ok": True, "all_passed": True,
                          "latency_ms_by_workload": {UUID: 5.0},
                          "clock_bracket_by_workload": {UUID: _wide_bracket()}},
        "v2_compile": {"ok": False, "error": "boom"},
    }
    proc, out = _build(tmp_path, {"SOLEXBENCH_CLOCK_BASIS": "unlocked"},
                       _tb_doc({}, variants), _sol_doc())
    assert proc.returncode == 0, proc.stderr
    w = json.loads(out.read_text())["problems"][KEY]["workloads"][UUID]
    assert w["t_b_ms"] == 5.0, "the fastest passing variant, as the runner picks"
    assert w["t_b_variant"] == "v4_contiguous"
    assert w["t_b_admitted_by_interval"] is True
    assert w["scoreable"] is True
    assert w["t_sol_interval_halfwidth_rel"] > 0.1


def test_a_variant_with_no_clock_samples_is_not_recovered(tmp_path):
    """Recovery admits width, never absence.

    A sibling workload with a good bracket keeps the artifact alive, so what is
    being observed is the recovery declining this one rather than the whole file
    being rejected.
    """
    other = "wl-1"
    variants = {"v1_eager": {
        "ok": True, "all_passed": True,
        "latency_ms_by_workload": {UUID: 9.0, other: 4.0},
        "clock_bracket_by_workload": {UUID: _bracket(None, None),
                                      other: _wide_bracket()}}}
    sol = {"problems": {KEY: {"workloads": {UUID: _bound(), other: _bound()}}}}
    proc, out = _build(tmp_path, {"SOLEXBENCH_CLOCK_BASIS": "unlocked"},
                       _tb_doc({}, variants), sol)
    assert proc.returncode == 0, proc.stderr
    wls = json.loads(out.read_text())["problems"][KEY]["workloads"]
    assert wls[UUID]["t_b_ms"] is None and wls[UUID]["scoreable"] is False
    assert wls[other]["t_b_ms"] == 4.0, "the sibling with a bracket is recovered"


def test_a_memory_bound_bound_has_a_zero_width_interval_in_the_manifest(tmp_path):
    """End to end, the property that says the two terms never got merged."""
    proc, out = _build(
        tmp_path, {"SOLEXBENCH_CLOCK_BASIS": "unlocked"},
        _tb_doc({UUID: {"variant": "v1", "t_b_ms": 5.0, **_wide_bracket()}}),
        _sol_doc(_bound(compute_cycles=1.0, memory_bytes=8_000_000)))
    assert proc.returncode == 0, proc.stderr
    w = json.loads(out.read_text())["problems"][KEY]["workloads"][UUID]
    assert w["t_sol_bottleneck_at_clock_min"] == "memory"
    assert w["t_sol_interval_halfwidth_rel"] == pytest.approx(0.0, abs=1e-12)


def test_the_locked_manifest_carries_no_interval_fields_at_all(tmp_path):
    """The frozen MI350X corpus must not move. Not a zero width -- absent."""
    proc, out = _build(
        tmp_path, {"SOLEXBENCH_CLOCK_BASIS": "locked",
                   "SOLEXBENCH_F_LOCK_MHZ": "1300"},
        _tb_doc({UUID: {"variant": "v1", "t_b_ms": 5.0}}), _sol_doc())
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text())
    w = doc["problems"][KEY]["workloads"][UUID]
    from solexbench_rocm.t_sol_at import INTERVAL_FIELDS
    assert not (set(INTERVAL_FIELDS) & set(w)), "locked records gained a field"
    assert "t_sol_interval_absent" not in w
    assert "t_sol_interval" not in doc
    assert "t_sol_interval_halfwidth_max" not in doc["problems"][KEY]
    assert w["t_sol_ms"] == 1e-3, "the locked bound is still the reference one"
