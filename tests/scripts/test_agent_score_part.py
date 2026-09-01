# SPDX-License-Identifier: Apache-2.0
"""`agent_score.py` refuses to pair one part's timings with another's bounds.

This driver re-times every surviving kernel on GPU 0 and scores those times
against a manifest. It had no part concept at all, and `--manifest` defaulted to
`artifacts/09/manifest-v1.json` -- MI350X's frozen release manifest. Run on the
MI355X node with no flags it therefore measured MI355X silicon and scored it
against MI350X bounds, silently. It wrote five of the seven scored runs on disk.

The direction is measured and it is the dangerous one: scoring the 2078
MI355X-measured records against the MI350X manifest raises `S` on 1996 of them,
mean 0.6377 -> 0.7214, and takes `t_k < T_SOL` violations from 12 to 273. A
score inflated this way is not detectable downstream -- the bound is simply the
wrong part's.

So the rule these tests pin is *fail closed*: resolve the part from every piece
of evidence there is, refuse when they disagree, and refuse when there is no
evidence at all. Never guess, and never default.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import agent_score as ags  # noqa: E402


def _prov(part: str | None) -> dict:
    """Provenance as the tree actually writes it: device names, no `part`.

    12,970 of 14,379 artifacts are attributable only this way, which is why the
    device-name fallback is load-bearing and is not being removed.
    """
    p: dict = {"utc": "2026-08-15T00:00:00+00:00", "git_sha": "deadbeef"}
    if part:
        p["torch"] = {"available": True, "device_count": 1,
                      "devices": [f"AMD Instinct {part}"]}
    return p


@pytest.fixture
def run(tmp_path):
    d = tmp_path / "run"
    (d / "retimed").mkdir(parents=True)
    (d / "run.json").write_text(json.dumps(
        {"run_id": "t", "model": "m", "harness": "codex", "sessions": {}}))
    return d


def _manifest(tmp_path, part: str | None, name: str = "manifest.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({"_provenance": _prov(part),
                             "manifest_version": "vT", "problems": {}}))
    return p


def _node(monkeypatch, detected: str | None, container: str | None = None):
    """What this process, and the measurement container, can see."""
    real = ags.detected_part
    monkeypatch.setattr(ags, "detected_part",
                        lambda devices=None: detected if devices is None
                        else real(devices))
    monkeypatch.setattr(ags, "_container_detected_part", lambda gpu: container)


def _main(monkeypatch, run: Path, manifest: Path, *extra: str) -> int:
    monkeypatch.setattr(sys, "argv", ["agent_score.py", "--run", str(run),
                                      "--manifest", str(manifest), *extra])
    return ags.main()


# -- the resolvers ----------------------------------------------------------

def test_part_claims_reads_both_conventions_and_the_evidence():
    """Union, never substitution: the top-level `part` is the only part check in
    the tree currently doing work, and `_provenance.part` is where the fix puts
    it. Reading one instead of the other kills a live guard."""
    doc = {"part": "MI355X", "_provenance": {"part": "MI355X",
                                             **_prov("MI355X")}}
    assert set(ags._part_claims(doc, "m").values()) == {"MI355X"}
    assert len(ags._part_claims(doc, "m")) == 3


def test_part_claims_reports_a_split():
    doc = {"part": "MI350X", "_provenance": _prov("MI355X")}
    part, err = ags._agree(ags._part_claims(doc, "manifest"), "the manifest")
    assert part is None and "more than one part" in err


def test_a_document_with_no_devices_does_not_borrow_this_hosts_cards(monkeypatch):
    """`detected_part(None)` asks the LOCAL cards. A manifest that names no
    device would otherwise be attributed to whatever node happened to read it --
    an inference dressed as the manifest's own statement."""
    _node(monkeypatch, detected="MI355X")
    assert ags._part_claims({"_provenance": {"utc": "x"}}, "manifest") == {}


def test_agree_on_no_claim_is_not_an_answer():
    part, err = ags._agree({}, "the manifest")
    assert part is None and "does not say" in err


def test_node_claims_uses_the_runs_own_re_times(run, monkeypatch):
    """The only *measured* claim about where a run's timings came from, and the
    one that makes `--reuse-retimed` resolvable on a host with no torch and no
    docker -- which is how five of the seven runs on disk were scored."""
    _node(monkeypatch, detected=None, container=None)
    (run / "retimed" / "L1__001.json").write_text(
        json.dumps({"_provenance": _prov("MI355X"), "per_workload": []}))
    assert ags.node_claims(0, None, run / "retimed") == {
        "existing retimed/": "MI355X"}


def test_node_claims_keeps_a_run_that_spans_two_parts_split(run, monkeypatch):
    _node(monkeypatch, detected=None, container=None)
    for i, part in enumerate(("MI350X", "MI355X")):
        (run / "retimed" / f"L1__00{i}.json").write_text(
            json.dumps({"_provenance": _prov(part)}))
    claims = ags.node_claims(0, None, run / "retimed")
    assert set(claims.values()) == {"MI350X", "MI355X"}
    assert ags._agree(claims, "this node")[0] is None


def test_node_claims_asks_the_container_when_this_python_has_no_torch(
        run, monkeypatch):
    _node(monkeypatch, detected=None, container="MI355X")
    assert ags.node_claims(0, None, run / "retimed") == {
        "measurement container": "MI355X"}


# -- fail closed ------------------------------------------------------------

def test_refuses_a_manifest_from_the_other_part(run, tmp_path, monkeypatch,
                                                capsys):
    _node(monkeypatch, detected="MI355X")
    rc = _main(monkeypatch, run, _manifest(tmp_path, "MI350X"))
    assert rc == 3
    err = capsys.readouterr().err
    assert "REFUSING" in err and "MI350X" in err and "MI355X" in err
    assert not (run / "scored.json").exists()


def test_refuses_when_the_node_cannot_be_resolved(run, tmp_path, monkeypatch,
                                                  capsys):
    """No torch here, no container, no prior re-time. A warning and a default is
    exactly how the wrong-part score would be produced."""
    _node(monkeypatch, detected=None, container=None)
    rc = _main(monkeypatch, run, _manifest(tmp_path, "MI355X"))
    assert rc == 3
    assert "does not say which part" in capsys.readouterr().err


def test_refuses_a_manifest_that_cannot_name_its_part(run, tmp_path,
                                                      monkeypatch, capsys):
    _node(monkeypatch, detected="MI355X")
    rc = _main(monkeypatch, run, _manifest(tmp_path, None))
    assert rc == 3
    assert "the manifest does not say" in capsys.readouterr().err


def test_a_declared_part_the_evidence_contradicts_is_refused(
        run, tmp_path, monkeypatch, capsys):
    _node(monkeypatch, detected="MI355X")
    rc = _main(monkeypatch, run, _manifest(tmp_path, "MI350X"), "--part", "MI350X")
    assert rc == 3
    assert "more than one part" in capsys.readouterr().err


def test_scores_when_the_part_agrees(run, tmp_path, monkeypatch):
    _node(monkeypatch, detected="MI355X")
    rc = _main(monkeypatch, run, _manifest(tmp_path, "MI355X"))
    assert rc == 0
    doc = json.loads((run / "scored.json").read_text())
    assert doc["part"] == "MI355X"
    assert doc["part_source"] == "detected"
    assert doc["part_claims"] == {"this process": "MI355X"}
    # Stated, not inferred, on the artifact this driver writes -- the whole
    # point of the change. `ingest.py` and the gates read this key first.
    assert doc["_provenance"]["part"] == "MI355X"


def test_a_declaration_carries_a_node_nothing_else_can_identify(
        run, tmp_path, monkeypatch):
    _node(monkeypatch, detected=None, container=None)
    rc = _main(monkeypatch, run, _manifest(tmp_path, "MI355X"), "--part", "MI355X")
    assert rc == 0
    doc = json.loads((run / "scored.json").read_text())
    assert (doc["part"], doc["part_source"]) == ("MI355X", "declared")


def test_a_fresh_re_time_from_the_wrong_card_stops_the_run(
        run, tmp_path, monkeypatch, capsys):
    """The one contradiction that can only arrive after the sweep starts: a
    human declares the node and the card says otherwise. The timings stay on
    disk -- only the aggregation is refused, and `--reuse-retimed` picks them
    up again against the right manifest."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "kernel.py").write_text("# kernel\n")
    (run / "run.json").write_text(json.dumps(
        {"run_id": "t", "model": "m", "harness": "codex",
         "sessions": {"L1__001": {"sandbox": str(sandbox)}}}))
    _node(monkeypatch, detected=None, container=None)
    monkeypatch.setattr(ags, "retime", lambda *a, **k: {
        "ok": True, "workloads": 0, "passed": 0, "per_workload": [],
        "_provenance": _prov("MI355X"),
        "tolerance_root": "/work/artifacts/05/workloads"})
    rc = _main(monkeypatch, run, _manifest(tmp_path, "MI350X"), "--part", "MI350X")
    assert rc == 4
    assert "measured on MI355X" in capsys.readouterr().err
    assert not (run / "scored.json").exists()


def test_there_is_no_manifest_default(monkeypatch, run):
    """The default WAS the defect: `artifacts/09/manifest-v1.json` is MI350X's."""
    monkeypatch.setattr(sys, "argv", ["agent_score.py", "--run", str(run)])
    with pytest.raises(SystemExit):
        ags.main()
    assert not hasattr(ags, "MANIFEST")
