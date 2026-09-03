# SPDX-License-Identifier: Apache-2.0
"""Freshness must not call an unreadable input tree fresh."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")


def _meta(signature: dict) -> dict:
    return {
        "input_signature": json.dumps(signature),
        "input_extra_roots": "[]",
        "input_manifest_path": "artifacts/09/manifest-v1.2.json",
        "part": "MI350X",
    }


def test_a_database_with_no_recorded_signature_is_unknown(app_mod):
    got = app_mod.freshness(_meta({}))
    assert got["stale"] is None
    assert "unknown" in got["error"]


def test_partially_missing_build_inputs_are_unknown_not_stale(app_mod, monkeypatch):
    recorded = {
        "n_files": 3,
        "total_bytes": 100,
        "max_mtime": 10.0,
        "newest_file": "artifacts/10/x/scored.json",
    }
    monkeypatch.setattr(
        app_mod.inputs,
        "signature",
        lambda *a, **k: {
            "n_files": 2,
            "total_bytes": 90,
            "max_mtime": 0.0,
            "newest_file": None,
        },
    )
    got = app_mod.freshness(_meta(recorded))
    assert got["stale"] is None
    assert "1 build input file(s) are unavailable" in got["reasons"][0]


def test_a_real_input_change_is_stale(app_mod, monkeypatch):
    recorded = {
        "n_files": 1,
        "total_bytes": 100,
        "max_mtime": 10.0,
        "newest_file": "old",
    }
    monkeypatch.setattr(
        app_mod.inputs,
        "signature",
        lambda *a, **k: {**recorded, "total_bytes": 101},
    )
    got = app_mod.freshness(_meta(recorded))
    assert got["stale"] is True
    assert got["reasons"] == ["an input file changed size since the last build"]


def test_unknown_freshness_is_rendered(client, board):
    board.write("UPDATE meta SET value='{}' WHERE key='input_signature'")
    page = client.get("/").text
    assert "Artifact freshness is unknown" in page
    assert "cannot prove" in page


def test_an_external_build_path_is_unknown_and_never_echoed(app_mod):
    meta = _meta({
        "n_files": 1, "total_bytes": 1, "max_mtime": 1,
        "newest_file": "<external>/run.json",
    })
    meta["input_paths_portable"] = "0"
    meta["input_extra_roots"] = '["/home/operator/private"]'
    got = app_mod.freshness(meta)
    assert got["stale"] is None
    assert got.get("rebuild_command") is None
    assert "/home/operator" not in json.dumps(got)


def test_retime_part_provenance_changes_the_input_signature(
        tmp_path, monkeypatch, app_mod):
    runs = tmp_path / "runs"
    retimed = runs / "run-a" / "retimed"
    retimed.mkdir(parents=True)
    (runs / "run-a" / "scored.json").write_text("{}")
    part = retimed / "problem.json"
    part.write_text('{"part":"MI350X"}')
    monkeypatch.setattr(app_mod.inputs, "AGENT_RUNS", runs)
    monkeypatch.setattr(app_mod.inputs, "CANDIDATES", tmp_path / "no-candidates")
    monkeypatch.setattr(app_mod.inputs, "AUTHORITATIVE", tmp_path / "no-authoritative")
    monkeypatch.setattr(app_mod.inputs, "DEFERRED", tmp_path / "no-deferred")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    before = app_mod.inputs.signature(manifest_path=manifest)
    part.write_text('{"part":"MI355X","changed":true}')
    after = app_mod.inputs.signature(manifest_path=manifest)
    assert before["n_files"] == after["n_files"] == 3
    assert before["total_bytes"] != after["total_bytes"]
    assert "<external>" in after["newest_file"]
