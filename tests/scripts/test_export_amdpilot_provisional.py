# SPDX-License-Identifier: Apache-2.0
"""The public provisional export is evidence, never an invented score."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import export_amdpilot_provisional as export  # noqa: E402


def _job(
    job_id: str,
    *,
    model: str | None = "GLM-5.2-local",
    problem: str = "L1__001_x",
    state: str = "succeeded",
    manifest: str = "v4",
    part: str = "MI355X",
    origin: str = "production",
    purpose: str = "benchmark",
    latest: bool = True,
) -> dict:
    submission = (
        {
            "count": 2,
            "latest": {
                "n": 2,
                "name": "0002-123",
                "files": ["kernel.py"],
                "note": "All 16 pass at 99x. This remains a local claim.",
                "ts": 20,
            },
        }
        if latest
        else {}
    )
    return {
        "job_id": job_id,
        "task_id": f"t-{job_id}",
        "task_name": f"solbench/{problem}",
        "state": state,
        "image": "amdpilotv2/kda-job:1",
        "created_at": 10,
        "last_update_at": 20,
        "payload": {
            "workflow": "kda",
            "origin": origin,
            "run_purpose": purpose,
            "model_under_test": model,
            "benchmark": {
                "problem_key": problem,
                "manifest_version": manifest,
                "manifest_measured_on": part,
            },
            "hardware": {"part": part},
        },
        "detail": {"harness": "kda", "submission": submission},
    }


def _kernel(job_id: str, origin_path: str | None) -> dict | None:
    if job_id == "j-no-source":
        return None
    origin_path = origin_path or "submissions/0001-fallback/kernel.py"
    source = f"# {job_id} {origin_path}\ndef run(x):\n    return x\n"
    return {
        "source": source,
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "bytes": len(source.encode()),
        "artifact_id": f"{job_id}-kernel",
        "origin_path": origin_path,
    }


def test_only_attributable_production_kda_jobs_are_exported():
    jobs = [
        _job("j-good"),
        _job("j-failed", state="failed"),
        _job("j-wrong-manifest", manifest="v3"),
        _job("j-wrong-part", part="MI350X"),
        _job("j-smoke", origin="smoketest"),
        _job("j-validation", purpose="pipeline-validation"),
        _job("j-no-model", model=None),
        _job("j-no-latest", latest=False),
        _job("j-no-source"),
        _job("j-good"),  # duplicate arrival from an overlapping page
    ]
    got = export.build_snapshot(
        jobs,
        _kernel,
        part="MI355X",
        manifest_version="v4",
        database_version="1.0.0+git.test",
        generated_at="2026-09-03T00:00:00+00:00",
    )

    assert [row["job_id"] for row in got["jobs"]] == [
        "j-good", "j-no-latest", "j-no-source"]
    assert got["jobs"][1]["submission"]["validation_note"] is None, (
        "a note about an absent latest submission must not be attached to an "
        "older retained kernel")
    assert got["jobs"][2]["evidence"] == "validation_note_only"
    assert got["jobs"][2]["kernel"] is None
    assert got["counts"]["jobs_exported"] == 3
    assert got["counts"]["excluded"] == {
        "manifest_mismatch": 1,
        "missing_or_duplicate_job_id": 1,
        "model_unattributed": 1,
        "non_benchmark": 1,
        "non_production": 1,
        "not_succeeded": 1,
        "part_mismatch": 1,
    }


def test_free_form_validation_never_becomes_a_score_or_rank():
    got = export.build_snapshot(
        [_job("j-claim")],
        _kernel,
        part="MI355X",
        manifest_version="v4",
        database_version="test",
        generated_at="now",
    )
    row = got["jobs"][0]
    assert "99x" in row["submission"]["validation_note"]
    assert not ({"score", "rank", "speedup", "passed"} & set(row))
    assert got["policy"] == {
        "evidence_tier": "provisional",
        "ranked": False,
        "score_source": None,
        "note": (
            "Validation notes are job-authored local measurements. They are "
            "preserved as text and never parsed into a leaderboard score."
        ),
    }


def test_all_jobs_are_kept_while_model_problem_counts_are_distinct():
    jobs = [
        _job("j-a", model="A", problem="L1__001_x"),
        _job("j-b", model="A", problem="L1__001_x"),
        _job("j-c", model="A", problem="L2__002_y"),
        _job("j-d", model="B", problem="L1__001_x"),
    ]
    got = export.build_snapshot(
        jobs,
        _kernel,
        part="MI355X",
        manifest_version="v4",
        database_version="test",
        generated_at="now",
    )
    assert len(got["jobs"]) == 4
    assert got["counts"]["jobs_by_model"] == {"A": 3, "B": 1}
    assert got["counts"]["problems_by_model"] == {"A": 2, "B": 1}


def test_write_atomic_round_trips(tmp_path):
    target = tmp_path / "nested" / "provisional.json"
    payload = {"schema": export.SCHEMA, "jobs": [{"job_id": "j-1"}]}
    export.write_atomic(target, payload)
    assert json.loads(target.read_text()) == payload
    assert not list(target.parent.glob(f".{target.name}.*"))


def test_job_paging_refuses_a_moving_store():
    calls = 0

    def moving(url: str) -> dict:
        nonlocal calls
        if url.endswith("/status"):
            return {"version": "test"}
        calls += 1
        return {
            "jobs": [_job("j-1")] if calls == 1 else [],
            "total": 1 if calls == 1 else 2,
        }

    with pytest.raises(RuntimeError, match="job count changed"):
        export.list_jobs("http://database", moving)
