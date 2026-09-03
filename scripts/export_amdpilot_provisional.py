#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export existing AMDPilot v2 KDA results without promoting them to scores.

This is a read-only import from the live control plane.  It deliberately does
not parse a job's free-form validation note into a score: those numbers were
measured by different in-job evaluators, not by ``scripts/agent_score.py`` on
an exclusive card.  The resulting ``provisional.json`` is a source for an
unranked section of the board, never a substitute for ``scored.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "10" / "amdpilot-v2-provisional" / "provisional.json"
)
SCHEMA = "amdpilot-provisional/1"
PAGE = 1000
USER_AGENT = "sol-execbench-rocm/amdpilot-provisional-export"

JsonFetch = Callable[[str], dict]
ArtifactLoad = Callable[[str, str | None], dict | None]


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def list_jobs(database_url: str, fetch: JsonFetch = fetch_json) -> tuple[list[dict], str]:
    """Read one stable newest-first window, paging until the store says done."""
    rows: list[dict] = []
    offset = 0
    expected_total = None
    base = database_url.rstrip("/")
    while expected_total is None or offset < expected_total:
        query = urllib.parse.urlencode(
            {"limit": PAGE, "offset": offset, "order": "desc", "order_by": "created"}
        )
        body = fetch(f"{base}/v1/jobs?{query}")
        batch = body.get("jobs")
        if not isinstance(batch, list):
            raise TypeError("Database /v1/jobs did not return a jobs list")
        advertised = body.get("total")
        if not isinstance(advertised, int):
            raise TypeError("Database /v1/jobs did not return an integer total")
        if expected_total is None:
            expected_total = advertised
        elif advertised != expected_total:
            raise RuntimeError(
                "job count changed during export; retry to avoid a shifted page"
            )
        rows.extend(batch)
        offset += len(batch)
        if not batch:
            break
    final = fetch(f"{base}/v1/jobs?limit=1&offset=0&order=desc&order_by=created")
    if final.get("total") != expected_total:
        raise RuntimeError(
            "job count changed during export; retry to produce one stable snapshot"
        )
    status = fetch(f"{base}/status")
    version = str(status.get("version") or "")
    return rows, version


def artifact_loader(overlay_url: str, fetch: JsonFetch = fetch_json) -> ArtifactLoad:
    """Return the latest kernel's held bytes, or None when no bytes were kept."""

    def load(job_id: str, origin_path: str | None) -> dict | None:
        root = overlay_url.rstrip("/")
        listing = fetch(f"{root}/api/db/jobs/{job_id}/artifacts")
        rows = ((listing.get("data") or {}).get("artifacts") or [])
        candidates = [
            row
            for row in rows
            if row.get("origin_family") == "kernel.py"
            and row.get("held") == "held"
            and row.get("artifact_id")
        ]
        match = next(
            (
                row
                for row in candidates
                if row.get("origin_path") == origin_path
            ),
            None,
        )
        if match is None and candidates:
            match = max(candidates, key=lambda row: str(row.get("origin_path") or ""))
        if match is None:
            return None
        artifact_id = urllib.parse.quote(str(match["artifact_id"]), safe="")
        content = fetch(
            f"{root}/api/db/jobs/{job_id}/artifacts/{artifact_id}/content"
        ).get("data") or {}
        source = content.get("text")
        if not isinstance(source, str):
            return None
        digest = hashlib.sha256(source.encode()).hexdigest()
        recorded = str(match.get("sha256") or content.get("sha256") or "")
        if recorded and digest != recorded:
            raise ValueError(
                f"{job_id} {origin_path}: source digest {digest} != {recorded}"
            )
        return {
            "source": source,
            "sha256": digest,
            "bytes": len(source.encode()),
            "artifact_id": str(match["artifact_id"]),
            "origin_path": str(match.get("origin_path") or ""),
        }

    return load


def iso_utc(value: object) -> str | None:
    try:
        return datetime.fromtimestamp(float(value), UTC).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def compact_note(value: object, limit: int = 2000) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text[:limit] if text else None


def _is_kda_sol(job: dict) -> bool:
    payload = job.get("payload") or {}
    detail = job.get("detail") or {}
    return (
        str(job.get("task_name") or "").startswith("solbench/")
        and (
            payload.get("workflow") == "kda"
            or payload.get("kind") == "kda"
            or detail.get("harness") == "kda"
            or "kda-job" in str(job.get("image") or "")
        )
    )


def build_snapshot(
    jobs: list[dict],
    load_artifact: ArtifactLoad,
    *,
    part: str,
    manifest_version: str,
    database_version: str,
    generated_at: str,
) -> dict:
    """Select attributable terminal KDA jobs and state their evidence depth."""
    excluded: Counter[str] = Counter()
    selected: list[dict] = []
    seen_ids: set[str] = set()
    kda_jobs_seen = 0

    for job in jobs:
        if not _is_kda_sol(job):
            continue
        kda_jobs_seen += 1
        if job.get("state") != "succeeded":
            excluded["not_succeeded"] += 1
            continue

        job_id = str(job.get("job_id") or "")
        if not job_id or job_id in seen_ids:
            excluded["missing_or_duplicate_job_id"] += 1
            continue
        seen_ids.add(job_id)

        payload = job.get("payload") or {}
        benchmark = payload.get("benchmark") or {}
        hardware = payload.get("hardware") or {}
        if benchmark.get("manifest_version") != manifest_version:
            excluded["manifest_mismatch"] += 1
            continue
        measured_part = (
            benchmark.get("manifest_measured_on") or hardware.get("part")
        )
        if measured_part != part:
            excluded["part_mismatch"] += 1
            continue
        if payload.get("origin") != "production":
            excluded["non_production"] += 1
            continue
        if payload.get("run_purpose") != "benchmark":
            excluded["non_benchmark"] += 1
            continue

        model = payload.get("model_under_test") or payload.get("model_requested")
        problem_key = benchmark.get("problem_key")
        detail = job.get("detail") or {}
        latest = ((detail.get("submission") or {}).get("latest") or {})
        submission_name = latest.get("name")
        files = latest.get("files") or []
        if not isinstance(model, str) or not model.strip():
            excluded["model_unattributed"] += 1
            continue
        if not isinstance(problem_key, str) or not problem_key:
            excluded["problem_unattributed"] += 1
            continue
        origin_path = (
            f"submissions/{submission_name}/kernel.py"
            if isinstance(submission_name, str) and "kernel.py" in files
            else None
        )
        kernel = load_artifact(job_id, origin_path)
        retained_submission = None
        if kernel is not None:
            retained_path = str(kernel.get("origin_path") or "")
            retained_parts = retained_path.split("/")
            retained_submission = (
                retained_parts[1]
                if len(retained_parts) == 3
                and retained_parts[0] == "submissions"
                and retained_parts[2] == "kernel.py"
                else None
            )
            if not retained_submission:
                raise ValueError(
                    f"{job_id}: retained kernel has unusable origin path "
                    f"{retained_path!r}"
                )
        has_latest = isinstance(submission_name, str) and bool(submission_name)
        has_note = compact_note(latest.get("note")) is not None
        if kernel is None and not (has_latest and has_note):
            excluded["no_public_result_evidence"] += 1
            continue
        matches_latest = retained_submission == submission_name
        evidence = (
            "kernel_and_validation_note"
            if kernel is not None and matches_latest and has_note
            else "kernel_source"
            if kernel is not None
            else "validation_note_only"
        )

        selected.append(
            {
                "job_id": job_id,
                "task_id": str(job.get("task_id") or "") or None,
                "task_name": str(job.get("task_name") or ""),
                "problem_key": problem_key,
                "model": model.strip(),
                "created_utc": iso_utc(job.get("created_at")),
                "finished_utc": iso_utc(
                    job.get("last_update_at") or latest.get("ts")
                ),
                "study": payload.get("study"),
                "arm": payload.get("arm"),
                "evidence": evidence,
                "submission": {
                    "n": latest.get("n") if matches_latest or kernel is None else None,
                    "name": retained_submission or submission_name,
                    "utc": (
                        iso_utc(latest.get("ts"))
                        if matches_latest or kernel is None
                        else None
                    ),
                    "validation_note": (
                        compact_note(latest.get("note"))
                        if matches_latest or kernel is None
                        else None
                    ),
                },
                "kernel": kernel,
                "provenance": {
                    "workflow": "kda",
                    "origin": "production",
                    "run_purpose": "benchmark",
                    "manifest_version": manifest_version,
                    "manifest_measured_on": part,
                    "dsl_brief_sha256": payload.get("dsl_brief_sha256"),
                    "image": job.get("image"),
                },
            }
        )

    selected.sort(
        key=lambda row: (
            row["model"].lower(),
            row["problem_key"],
            row["created_utc"] or "",
            row["job_id"],
        )
    )
    by_model = Counter(row["model"] for row in selected)
    evidence = Counter(row["evidence"] for row in selected)
    problems_by_model = {
        model: len({row["problem_key"] for row in selected if row["model"] == model})
        for model in by_model
    }
    return {
        "schema": SCHEMA,
        "part": part,
        "manifest_version": manifest_version,
        "generated_at": generated_at,
        "source": {
            "system": "amdpilot-v2",
            "component": "database",
            "version": database_version,
            "selection": "terminal succeeded production KDA SOL jobs",
        },
        "policy": {
            "evidence_tier": "provisional",
            "ranked": False,
            "score_source": None,
            "note": (
                "Validation notes are job-authored local measurements. They are "
                "preserved as text and never parsed into a leaderboard score."
            ),
        },
        "counts": {
            "jobs_read": len(jobs),
            "kda_jobs_seen": kda_jobs_seen,
            "jobs_exported": len(selected),
            "models": len(by_model),
            "evidence": dict(sorted(evidence.items())),
            "jobs_by_model": dict(sorted(by_model.items())),
            "problems_by_model": dict(sorted(problems_by_model.items())),
            "excluded": dict(sorted(excluded.items())),
        },
        "jobs": selected,
    }


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default="http://127.0.0.1:7205")
    parser.add_argument("--overlay-url", default="http://127.0.0.1:7100")
    parser.add_argument("--part", default="MI355X")
    parser.add_argument("--manifest-version", default="v4")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    jobs, version = list_jobs(args.database_url)
    snapshot = build_snapshot(
        jobs,
        artifact_loader(args.overlay_url),
        part=args.part,
        manifest_version=args.manifest_version,
        database_version=version,
        generated_at=datetime.now(UTC).isoformat(),
    )
    write_atomic(args.output, snapshot)
    print(
        f"{args.output}: {snapshot['counts']['jobs_exported']} jobs, "
        f"{snapshot['counts']['models']} models"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
