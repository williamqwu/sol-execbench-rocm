# SPDX-License-Identifier: Apache-2.0
"""The authoritative scorer must use the tolerance tree for its GPU part.

Bounds and correctness tolerances are both part-specific.  A score is invalid
if it pairs MI355X timings and bounds with the older MI350X tolerance tree, so
the selected tree must be explicit, stamped, and checked before re-use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import agent_score as ags  # noqa: E402
import retime_parallel as rp  # noqa: E402
from tolerance_roots import (  # noqa: E402
    container_tolerance_root,
    recorded_tolerance_root,
)


@pytest.mark.parametrize("part,want", [
    ("MI350X", "/work/artifacts/05/workloads"),
    ("MI355X", "/work/artifacts/05-MI355X/workloads"),
])
def test_part_selects_its_own_tolerance_tree(part, want):
    assert container_tolerance_root(part) == want


def test_an_unknown_part_has_no_guessed_tolerance_tree():
    with pytest.raises(ValueError, match="no tolerance tree"):
        container_tolerance_root("MI400X")


def test_only_a_provenanced_legacy_mi350x_retime_has_an_inferable_tree():
    legacy = {"_provenance": {"task": "10-agent-eval"}}

    assert recorded_tolerance_root(legacy, "MI350X") == \
        "/work/artifacts/05/workloads"
    assert recorded_tolerance_root(legacy, "MI355X") is None
    assert recorded_tolerance_root({}, "MI350X") is None


def _runner_that_writes_the_requested_artifact(seen: dict):
    def run(cmd, *, env, **kwargs):
        seen["root"] = env["SOLEXBENCH_WORKLOADS_ROOT"]
        out = Path(cmd[cmd.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "ok": True,
            "workloads": 0,
            "passed": 1,
            "per_workload": [],
            "_provenance": {"part": "MI355X"},
        }))
        return SimpleNamespace(returncode=0, stderr="")

    return run


def test_serial_retime_uses_and_stamps_the_selected_tree(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ags, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(ags, "_await_exclusive_card", lambda gpu: ([], 0.0))
    monkeypatch.setattr(ags.subprocess, "run",
                        _runner_that_writes_the_requested_artifact(seen))
    out = tmp_path / "retimed.json"

    got = ags.retime("L1__001", tmp_path / "kernel.py", out, 0, 50, 10,
                     2400, "/work/artifacts/05-MI355X/workloads")

    assert seen["root"] == "/work/artifacts/05-MI355X/workloads"
    assert got["tolerance_root"] == seen["root"]
    assert json.loads(out.read_text())["tolerance_root"] == seen["root"]


def test_parallel_retime_uses_and_stamps_the_selected_tree(tmp_path,
                                                            monkeypatch):
    seen = {}
    monkeypatch.setattr(rp, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(rp, "foreign_on", lambda gpu: [])
    monkeypatch.setattr(rp.subprocess, "run",
                        _runner_that_writes_the_requested_artifact(seen))
    out = tmp_path / "retimed.json"

    assert rp.measure("L1__001", "/tmp/kernel.py", out, 0, 50, 10, 2400,
                      8, "/work/artifacts/05-MI355X/workloads")
    assert seen["root"] == "/work/artifacts/05-MI355X/workloads"
    assert json.loads(out.read_text())["tolerance_root"] == seen["root"]


def _manifest(tmp_path: Path, part: str = "MI355X") -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "_provenance": {"part": part},
        "manifest_version": "test",
        "problems": {},
    }))
    return path


def _run_with_one_retime(tmp_path: Path, tolerance_root=...,
                         part: str = "MI355X") -> Path:
    run = tmp_path / "run"
    sandbox = tmp_path / "sandbox"
    (run / "retimed").mkdir(parents=True)
    sandbox.mkdir()
    (sandbox / "kernel.py").write_text("# kernel\n")
    (run / "run.json").write_text(json.dumps({
        "run_id": "test",
        "model": "m",
        "harness": "codex",
        "sessions": {"L1__001": {"sandbox": str(sandbox)}},
    }))
    artifact = {
        "ok": True,
        "workloads": 0,
        "passed": 0,
        "per_workload": [],
        "_provenance": {"task": "10-agent-eval", "part": part},
    }
    if tolerance_root is not ...:
        artifact["tolerance_root"] = tolerance_root
    (run / "retimed" / "L1__001.json").write_text(json.dumps(artifact))
    return run


@pytest.mark.parametrize("recorded", [
    pytest.param(..., id="missing-stamp"),
    pytest.param("/work/artifacts/05/workloads", id="wrong-part-tree"),
])
def test_reuse_refuses_unverifiable_or_wrong_tolerance_tree(
        tmp_path, monkeypatch, capsys, recorded):
    run = _run_with_one_retime(tmp_path, recorded)
    monkeypatch.setattr(ags, "detected_part",
                        lambda devices=None: None if devices is None else "MI355X")
    monkeypatch.setattr(ags, "_container_detected_part", lambda gpu: None)
    monkeypatch.setattr(sys, "argv", [
        "agent_score.py", "--run", str(run), "--manifest",
        str(_manifest(tmp_path)), "--part", "MI355X", "--reuse-retimed",
    ])

    assert ags.main() == 5
    err = capsys.readouterr().err
    assert "REFUSING to reuse" in err
    assert "tolerance_root" in err
    assert not (run / "scored.json").exists()


def test_reuse_accepts_a_retime_from_the_same_tolerance_tree(
        tmp_path, monkeypatch):
    root = "/work/artifacts/05-MI355X/workloads"
    run = _run_with_one_retime(tmp_path, root)
    monkeypatch.setattr(ags, "detected_part",
                        lambda devices=None: None if devices is None else "MI355X")
    monkeypatch.setattr(ags, "_container_detected_part", lambda gpu: None)
    monkeypatch.setattr(ags, "retime",
                        lambda *args, **kwargs: pytest.fail("must reuse"))
    monkeypatch.setattr(sys, "argv", [
        "agent_score.py", "--run", str(run), "--manifest",
        str(_manifest(tmp_path)), "--part", "MI355X", "--reuse-retimed",
    ])

    assert ags.main() == 0
    assert json.loads((run / "scored.json").read_text())["tolerance_root"] == root


def test_reuse_accepts_an_unstamped_legacy_mi350x_retime(
        tmp_path, monkeypatch):
    run = _run_with_one_retime(tmp_path, part="MI350X")
    monkeypatch.setattr(ags, "detected_part",
                        lambda devices=None: None if devices is None else "MI350X")
    monkeypatch.setattr(ags, "_container_detected_part", lambda gpu: None)
    monkeypatch.setattr(ags, "retime",
                        lambda *args, **kwargs: pytest.fail("must reuse"))
    monkeypatch.setattr(sys, "argv", [
        "agent_score.py", "--run", str(run), "--manifest",
        str(_manifest(tmp_path, "MI350X")), "--part", "MI350X",
        "--reuse-retimed",
    ])

    assert ags.main() == 0
    assert json.loads((run / "scored.json").read_text())["tolerance_root"] == \
        "/work/artifacts/05/workloads"


def test_parallel_only_missing_reuses_an_unstamped_legacy_mi350x_retime(
        tmp_path, monkeypatch):
    run = _run_with_one_retime(tmp_path, part="MI350X")
    monkeypatch.setattr(rp, "ROOT", tmp_path)
    monkeypatch.setattr(rp, "measure",
                        lambda *args, **kwargs: pytest.fail("must skip"))
    monkeypatch.setattr(sys, "argv", [
        "retime_parallel.py", "--run", str(run), "--part", "MI350X",
        "--gpus", "0", "--only-missing",
    ])

    assert rp.main() == 0


@pytest.mark.parametrize("existing", [
    pytest.param(None, id="unverifiable-mi355x"),
    pytest.param([], id="non-object-json"),
])
def test_parallel_only_missing_retimes_an_unverifiable_artifact(
        tmp_path, monkeypatch, existing):
    run = _run_with_one_retime(tmp_path)
    if existing is not None:
        (run / "retimed" / "L1__001.json").write_text(json.dumps(existing))
    seen = []
    monkeypatch.setattr(rp, "ROOT", tmp_path)
    monkeypatch.setattr(
        rp, "measure",
        lambda key, *args, **kwargs: seen.append(key) or True,
    )
    monkeypatch.setattr(sys, "argv", [
        "retime_parallel.py", "--run", str(run), "--part", "MI355X",
        "--gpus", "0", "--only-missing",
    ])

    assert rp.main() == 0
    assert seen == ["L1__001"]


def test_scored_artifact_stamps_the_selected_tolerance_tree(tmp_path,
                                                            monkeypatch):
    run = tmp_path / "run"
    (run / "retimed").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({
        "run_id": "test", "model": "m", "harness": "codex", "sessions": {}}))
    monkeypatch.setattr(ags, "detected_part",
                        lambda devices=None: "MI355X" if devices is None else None)
    monkeypatch.setattr(sys, "argv", [
        "agent_score.py", "--run", str(run), "--manifest",
        str(_manifest(tmp_path)),
    ])

    assert ags.main() == 0
    assert json.loads((run / "scored.json").read_text())["tolerance_root"] == \
        "/work/artifacts/05-MI355X/workloads"
