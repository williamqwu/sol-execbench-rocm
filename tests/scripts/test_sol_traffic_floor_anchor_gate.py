# SPDX-License-Identifier: Apache-2.0
"""The traffic tier's own gate: which anchors it runs against, and what a
rejected record is allowed to say.

Two defects, one artifact.

**The gate ran against the wrong tree.** `artifacts/03-MI355X/t_sol_traffic.json`
was built with `--t-b artifacts/06-MI355X/authoritative` while every MI355X
manifest declares `sources.t_b = artifacts/06-MI355X/authoritative-merged` and
is built from it. 237 records therefore shipped stamped `gated_against_t_b`
against anchors no published score uses, and `L1__057`/`650d87fb` shipped a tier
bound of 0.17040 ms against a published anchor of 0.10150 ms -- 1.68x the
measured time, inside the artifact whose own `--t-b` help says such a bound "is
rejected, not shipped". Nothing downstream was wrong, because `build_manifest`
re-applies the gate; the point is that the manifest was the safety net and this
tier's gate was decorative.

The remedy is a refusal, not a better default. A default is overridden by any
caller who passes the flag they have always passed -- which is precisely how
this happened, `docs/TODO-MI355X.md` step 9 still spelling the older tree.

**A rejected record carried a millisecond column with no clock.** 21 of them,
each with `t_sol_ms` and no `f_ref_mhz`, in a file whose header asserts a single
distinct per-record clock -- an invariant that held only because those 21 were
excluded from the set it is computed over. That is D63 in miniature and the
exact shape `t_sol_at.bound_ms` exists to refuse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import sol_traffic_floor  # noqa: E402

#: The simplest problem the tier can price: one streamed input, one output.
PLAIN = {
    "axes": {"n": {"type": "var"}},
    "inputs": {"x": {"shape": ["n"], "dtype": "float32"}},
    "outputs": {"y": {"shape": ["n"], "dtype": "float32"}},
    "reference": "def run(x):\n    return x + 1\n",
}

MERGED = "artifacts/06-MI355X/authoritative-merged"
OLD = "artifacts/06-MI355X/authoritative"
TIER = "artifacts/03-MI355X/t_sol_traffic.json"


def _tree(root: Path, *, t_b_ms: dict[str, float] | None = None) -> None:
    """A miniature repo: one problem, two anchor trees, one manifest.

    The two anchor trees hold the SAME anchors unless a test says otherwise,
    so a difference in outcome can only come from which tree was consulted --
    not from the numbers in it.
    """
    prob = root / "data" / "L1" / "057_plain"
    prob.mkdir(parents=True)
    (prob / "definition.json").write_text(json.dumps(PLAIN))
    (prob / "workload.jsonl").write_text(
        json.dumps({"uuid": "keep", "axes": {"n": 1_000_000}, "inputs": {}})
        + "\n"
        + json.dumps({"uuid": "drop", "axes": {"n": 1_000_000}, "inputs": {}})
        + "\n")

    anchors = t_b_ms if t_b_ms is not None else {"keep": 1000.0, "drop": 1e-6}
    for tree in (OLD, MERGED):
        d = root / tree
        d.mkdir(parents=True)
        (d / "L1__057_plain.json").write_text(json.dumps(
            {"winner_by_workload": {u: {"t_b_ms": v}
                                    for u, v in anchors.items()}}))

    (root / "arch.yaml").write_text(
        "freq_GHz: 2.4\nDRAM_byte_per_cycle: 3333.3\n")
    (root / "t_sol.json").write_text(json.dumps({"problems": {}}))

    man = root / "artifacts" / "09-MI355X"
    man.mkdir(parents=True)
    (man / "manifest-v4.json").write_text(json.dumps(
        {"sources": {"t_b": MERGED, "t_sol_traffic": TIER,
                     "t_sol": "artifacts/03-MI355X/t_sol.json"}}))


def _run(root: Path, monkeypatch, *t_b_and_flags: str) -> tuple[int, dict]:
    """`main()` against *root*, returning (exit code, payload written)."""
    written: dict = {}

    # Mirrors `provenance.write_artifact` exactly, `part` included: a stub that
    # swallowed **kwargs would go on passing while the call site drifted.
    def _capture(path, task, payload, extra_provenance=None, *,
                 part=None, allow_cross_part=False):
        written.update(payload)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload))
        return Path(path)

    monkeypatch.setattr(sol_traffic_floor, "write_artifact", _capture)
    monkeypatch.setattr(sol_traffic_floor, "ROOT", root)
    monkeypatch.setattr(sys, "argv", [
        "sol_traffic_floor.py",
        "--data", str(root / "data"), "--arch", str(root / "arch.yaml"),
        "--t-sol", str(root / "t_sol.json"), "--out", str(root / TIER),
        *t_b_and_flags])
    return sol_traffic_floor.main(), written


# -- which tree the gate ran against ----------------------------------------


def test_the_declared_tree_is_read_off_the_manifests_built_from_this_tier(
        tmp_path):
    _tree(tmp_path)
    assert sol_traffic_floor.declared_anchor_trees(
        tmp_path / TIER, tmp_path) == {"artifacts/09-MI355X/manifest-v4.json":
                                       MERGED}


def test_a_manifest_that_names_another_tier_constrains_nothing(tmp_path):
    """Two parts share this script. An MI350X manifest must not decide what an
    MI355X tier build is allowed to gate against."""
    _tree(tmp_path)
    other = tmp_path / "artifacts" / "09"
    other.mkdir(parents=True)
    (other / "manifest-v1.json").write_text(json.dumps(
        {"sources": {"t_b": "artifacts/06/authoritative",
                     "t_sol_traffic": "artifacts/03/t_sol_traffic.json"}}))
    assert set(sol_traffic_floor.declared_anchor_trees(
        tmp_path / TIER, tmp_path)) == {"artifacts/09-MI355X/manifest-v4.json"}


def test_a_manifest_with_no_sources_block_constrains_nothing(tmp_path):
    """Both frozen MI350X manifests state `sources: null`. They must not turn
    an MI350X tier build red -- a new gate that fails on the artifacts it was
    not written about is a regression, not a check."""
    _tree(tmp_path)
    man = tmp_path / "artifacts" / "09-MI355X"
    (man / "manifest-v1.json").write_text(json.dumps({"sources": None}))
    assert sol_traffic_floor.declared_anchor_trees(
        tmp_path / TIER, tmp_path) == {"artifacts/09-MI355X/manifest-v4.json":
                                       MERGED}


def test_a_candidate_is_not_a_manifest(tmp_path):
    """`candidate-*.json` is an experiment and is allowed to disagree."""
    _tree(tmp_path)
    man = tmp_path / "artifacts" / "09-MI355X"
    (man / "candidate-v3-gatefix.json").write_text(json.dumps(
        {"sources": {"t_b": OLD, "t_sol_traffic": TIER}}))
    assert sol_traffic_floor.declared_anchor_trees(
        tmp_path / TIER, tmp_path) == {"artifacts/09-MI355X/manifest-v4.json":
                                       MERGED}


def test_an_unreadable_manifest_is_reported_not_skipped(tmp_path, capsys):
    """The value of this function is that it refuses. A bare `except` would
    make it stop refusing exactly when the artifacts are in the worst shape,
    so the read failure has to reach a human."""
    _tree(tmp_path)
    (tmp_path / "artifacts" / "09-MI355X" / "manifest-v9.json").write_text("{")
    sol_traffic_floor.declared_anchor_trees(tmp_path / TIER, tmp_path)
    err = capsys.readouterr().err
    assert "manifest-v9.json" in err and "NOT being checked" in err


# -- the refusal ------------------------------------------------------------


def test_the_wrong_anchor_tree_is_refused_and_nothing_is_written(
        tmp_path, monkeypatch):
    """The shipped defect, reproduced: `--t-b .../authoritative` against a
    manifest built from `.../authoritative-merged`."""
    _tree(tmp_path)
    code, written = _run(tmp_path, monkeypatch, "--t-b", str(tmp_path / OLD))
    assert code == 2
    assert written == {}
    assert not (tmp_path / TIER).exists()


def test_the_declared_anchor_tree_builds(tmp_path, monkeypatch):
    _tree(tmp_path)
    code, written = _run(tmp_path, monkeypatch, "--t-b", str(tmp_path / MERGED))
    assert code == 0
    assert written["t_b"] == str(tmp_path / MERGED)
    assert written["t_b_manifest_mismatch"] is None


def test_the_mismatch_can_be_taken_on_purpose_and_is_then_recorded(
        tmp_path, monkeypatch):
    """Adopting a new anchor tree is legitimate and happens before the manifest
    that will name it exists. The escape hatch is explicit, and the artifact
    then STATES that it was gated against a tree a manifest disagrees with --
    an override that left no trace would just be the old default again."""
    _tree(tmp_path)
    code, written = _run(tmp_path, monkeypatch, "--t-b", str(tmp_path / OLD),
                         "--allow-anchor-mismatch")
    assert code == 0
    assert written["t_b"] == str(tmp_path / OLD)
    assert written["t_b_manifest_mismatch"] == {
        "artifacts/09-MI355X/manifest-v4.json": MERGED}


def test_the_gate_actually_changes_the_verdict_it_guards(tmp_path, monkeypatch):
    """Not merely that the trees differ -- that consulting the wrong one ships
    a record the right one rejects. This is `L1__057`/`650d87fb` in miniature:
    one workload whose bound is above its merged-tree anchor and below its
    older one."""
    _tree(tmp_path, t_b_ms={"keep": 1000.0, "drop": 1000.0})
    merged = tmp_path / MERGED / "L1__057_plain.json"
    merged.write_text(json.dumps({"winner_by_workload": {
        "keep": {"t_b_ms": 1000.0}, "drop": {"t_b_ms": 1e-6}}}))

    code, loose = _run(tmp_path, monkeypatch, "--t-b", str(tmp_path / OLD),
                       "--allow-anchor-mismatch")
    assert code == 0
    assert set(loose["problems"]["L1__057_plain"]["workloads"]) == {"keep",
                                                                   "drop"}

    code, strict = _run(tmp_path, monkeypatch, "--t-b", str(tmp_path / MERGED))
    assert code == 0
    assert set(strict["problems"]["L1__057_plain"]["workloads"]) == {"keep"}
    assert [r["workload"] for r in strict["rejected"]] == ["drop"]


# -- a rejected record says which clock its milliseconds are at -------------


def test_a_rejected_record_carries_the_clock_its_ms_column_is_at(
        tmp_path, monkeypatch):
    _tree(tmp_path)
    code, written = _run(tmp_path, monkeypatch, "--t-b", str(tmp_path / MERGED))
    assert code == 0
    assert written["f_ref_mhz"] == 2400.0
    assert written["rejected"], "the fixture must produce a rejection"
    for rec in written["rejected"]:
        assert rec["t_sol_ms"] is not None
        assert rec["f_ref_mhz"] == written["f_ref_mhz"]


def test_the_headers_one_clock_invariant_now_covers_the_rejected_list(
        tmp_path, monkeypatch):
    """The header claims a single distinct per-record clock. Before this, that
    held only because the rejected list was excluded from the set it is
    computed over -- so a reader would attach the header's clock to records
    that never said one, which is D63's mechanism exactly."""
    _tree(tmp_path)
    _, written = _run(tmp_path, monkeypatch, "--t-b", str(tmp_path / MERGED))
    bodies = [w["f_ref_mhz"]
              for p in written["problems"].values()
              for w in p["workloads"].values()]
    assert {*bodies, *(r["f_ref_mhz"] for r in written["rejected"])} == {
        written["f_ref_mhz"]}
