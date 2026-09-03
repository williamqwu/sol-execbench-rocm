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


def test_missing_build_inputs_are_unknown_not_fresh(app_mod, monkeypatch):
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
            "n_files": 0,
            "total_bytes": 0,
            "max_mtime": 0.0,
            "newest_file": None,
        },
    )
    got = app_mod.freshness(_meta(recorded))
    assert got["stale"] is None
    assert "not present" in got["reasons"][0]


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
