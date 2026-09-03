# SPDX-License-Identifier: Apache-2.0
"""The provisional artifact has its own fail-closed ingest path."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "leaderboard"))

import ingest  # noqa: E402


def _job(job_id: str, utc: str) -> dict:
    source = f"# {job_id}\ndef run(x):\n    return x\n"
    return {
        "job_id": job_id,
        "task_id": "t-1",
        "task_name": "solbench/L1__001_x",
        "problem_key": "L1__001_x",
        "model": "GLM-5.2-local",
        "created_utc": utc,
        "finished_utc": utc,
        "study": "test",
        "arm": "A",
        "evidence": "kernel_and_validation_note",
        "submission": {
            "n": 1,
            "name": "0001",
            "utc": utc,
            "validation_note": "Local validation only.",
        },
        "kernel": {
            "source": source,
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "bytes": len(source.encode()),
            "artifact_id": f"{job_id}-kernel",
        },
        "provenance": {
            "workflow": "kda",
            "origin": "production",
            "run_purpose": "benchmark",
            "manifest_version": "v4",
            "manifest_measured_on": "MI355X",
            "dsl_brief_sha256": "a" * 64,
            "image": "amdpilotv2/kda-job:1",
        },
    }


def _snapshot(*jobs: dict) -> dict:
    job_rows = list(jobs)
    models = {job["model"] for job in job_rows}
    evidence = {}
    for job in job_rows:
        evidence[job["evidence"]] = evidence.get(job["evidence"], 0) + 1
    return {
        "schema": ingest.PROVISIONAL_SCHEMA,
        "part": "MI355X",
        "manifest_version": "v4",
        "generated_at": "2026-09-03T00:00:00+00:00",
        "source": {
            "system": "amdpilot-v2",
            "component": "database",
            "version": "test",
            "selection": "terminal succeeded production KDA SOL jobs",
        },
        "policy": {
            "evidence_tier": "provisional",
            "ranked": False,
            "score_source": None,
            "note": "Local validation remains text.",
        },
        "counts": {
            "jobs_read": len(job_rows),
            "kda_jobs_seen": len(job_rows),
            "jobs_exported": len(job_rows),
            "models": len(models),
            "evidence": evidence,
            "excluded": {},
            "jobs_by_model": {
                model: sum(job["model"] == model for job in job_rows)
                for model in models
            },
            "problems_by_model": {
                model: len({
                    job["problem_key"] for job in job_rows
                    if job["model"] == model
                })
                for model in models
            },
        },
        "jobs": job_rows,
    }


def _db(tmp_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "board.db")
    conn.row_factory = sqlite3.Row
    conn.executescript((ROOT / "leaderboard" / "schema.sql").read_text())
    conn.execute(
        """INSERT INTO problem
           (key,category,name,n_workloads,n_scoreable,deferred)
           VALUES ('L1__001_x','L1','x',1,1,0)"""
    )
    return conn


def test_ingest_keeps_all_jobs_and_selects_the_newest_source(tmp_path):
    path = tmp_path / "provisional.json"
    path.write_text(json.dumps(_snapshot(
        _job("j-old", "2026-09-01T00:00:00+00:00"),
        _job("j-new", "2026-09-02T00:00:00+00:00"),
    )))
    conn = _db(tmp_path)
    assert ingest.ingest_provisional(
        conn, {"manifest_version": "v4"}, "MI355X", path
    ) == 2

    jobs = list(conn.execute(
        "SELECT job_id,selected FROM provisional_job ORDER BY job_id"))
    assert [(row["job_id"], row["selected"]) for row in jobs] == [
        ("j-new", 1), ("j-old", 0)]
    source = conn.execute(
        "SELECT kernel_source FROM provisional_job WHERE selected=1"
    ).fetchone()[0]
    assert source.startswith("# j-new")
    submission = conn.execute(
        "SELECT kind,board_visible,part FROM submission").fetchone()
    assert tuple(submission) == ("provisional", 0, "MI355X")
    assert conn.execute("SELECT COUNT(*) FROM result").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM run_kernel").fetchone()[0] == 0


def test_ingest_refuses_unpublished_job_fields(tmp_path):
    job = _job("j-1", "2026-09-01T00:00:00+00:00")
    job["env"] = {"SECRET": "must-not-travel"}
    path = tmp_path / "provisional.json"
    path.write_text(json.dumps(_snapshot(job)))
    conn = _db(tmp_path)
    with pytest.raises(SystemExit, match="unpublished keys.*env"):
        ingest.ingest_provisional(
            conn, {"manifest_version": "v4"}, "MI355X", path
        )


@pytest.mark.parametrize("where", ["document", "source", "policy"])
def test_ingest_refuses_unpublished_public_metadata(tmp_path, where):
    doc = _snapshot(_job("j-1", "2026-09-01T00:00:00+00:00"))
    target = doc if where == "document" else doc[where]
    target["private_path"] = "/home/operator/internal"
    path = tmp_path / "provisional.json"
    path.write_text(json.dumps(doc))
    conn = _db(tmp_path)
    with pytest.raises(SystemExit, match="unpublished keys.*private_path"):
        ingest.ingest_provisional(
            conn, {"manifest_version": "v4"}, "MI355X", path
        )


def test_note_only_evidence_is_kept_without_inventing_a_kernel(tmp_path):
    job = _job("j-note", "2026-09-01T00:00:00+00:00")
    job["kernel"] = None
    job["evidence"] = "validation_note_only"
    path = tmp_path / "provisional.json"
    path.write_text(json.dumps(_snapshot(job)))
    conn = _db(tmp_path)
    assert ingest.ingest_provisional(
        conn, {"manifest_version": "v4"}, "MI355X", path
    ) == 1
    assert conn.execute("SELECT selected FROM provisional_job").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM run_kernel").fetchone()[0] == 0


def test_ingest_refuses_a_manifest_mismatch(tmp_path):
    path = tmp_path / "provisional.json"
    path.write_text(json.dumps(_snapshot(
        _job("j-1", "2026-09-01T00:00:00+00:00")
    )))
    conn = _db(tmp_path)
    with pytest.raises(SystemExit, match="manifest 'v4'.*'v5'"):
        ingest.ingest_provisional(
            conn, {"manifest_version": "v5"}, "MI355X", path
        )


def test_snapshot_is_selected_only_for_its_own_part(tmp_path):
    path = tmp_path / "provisional.json"
    path.write_text(json.dumps(_snapshot()))
    assert ingest.provisional_path_for("MI355X", path) == path
    assert ingest.provisional_path_for("MI350X", path) is None


def test_public_input_locators_never_expose_an_external_root(tmp_path):
    assert ingest.repo_input_locator(ROOT / "artifacts" / "x.json") == \
        "artifacts/x.json"
    assert ingest.repo_input_locator(tmp_path / "private.json") is None


def test_ingest_meta_publishes_only_relocatable_input_paths(tmp_path):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    manifest = {
        "_provenance": {"torch": {"devices": ["AMD Instinct MI355X"]}},
        "manifest_version": "v4",
        "methodology": "test",
        "score_formula": "test",
        "problem_set": {
            "total_in_dataset": 0,
            "scoreable_problems": 0,
            "expected_by_category": {},
        },
        "stats": {"scoreable_workloads": 0},
        "problems": {},
    }
    ingest.ingest_meta(
        conn,
        manifest,
        "MI355X",
        [tmp_path / "private-runs"],
        manifest_path=ROOT / "artifacts" / "09-MI355X" / "manifest-v4.json",
        provisional_path=(
            ROOT / "artifacts" / "10" / "amdpilot-v2-provisional"
            / "provisional.json"
        ),
    )
    meta = dict(conn.execute("SELECT key,value FROM meta"))
    assert meta["input_manifest_path"] == \
        "artifacts/09-MI355X/manifest-v4.json"
    assert meta["input_provisional_path"] == \
        "artifacts/10/amdpilot-v2-provisional/provisional.json"
    assert meta["input_extra_roots"] == "[]"
    assert meta["input_external_root_count"] == "1"
    assert meta["input_paths_portable"] == "0"
    assert "/home/" not in json.dumps(meta)
