#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The write side: accept a kernel, queue it, report on it.

This is the only part of the service that is not read-only, and it is
deliberately the *smallest* part. Accepting a submission here does three
things: authenticate the caller, check the request is well formed, and put a
row in the queue. It does not run anything.

**What this is not.** Scoring a submission means executing code somebody else
wrote, on a GPU, in this repository's container. `env/solb` is a reproducibility
boundary, not a security one: it runs as a normal user with the repo
bind-mounted read-write and no seccomp profile, syscall filter or network
namespace of its own. A submitted kernel can read and write the tree.

So the trust model is: **authenticated internal users, and no one else.** The
token gates who can queue work; nothing gates what queued code can do once it
runs. That is acceptable for a team-internal service and is not acceptable for
anything public, and the difference is a sandbox that does not exist yet.
Stated here rather than in a design document because this is the file someone
will read before exposing the port.

Tokens live in a file, one `token:name` per line, path from
`SOLBENCH_SUBMIT_TOKENS` (default `leaderboard/.tokens`, gitignored). No token
file means the write API is disabled outright -- a service that accepts
anonymous submissions because its config is missing is worse than one that
refuses everything.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QUEUE_DB = Path(os.environ.get("SOLBENCH_QUEUE_DB", HERE / "queue.db"))
TOKENS = Path(os.environ.get("SOLBENCH_SUBMIT_TOKENS", HERE / ".tokens"))

# A kernel is a source file, not a payload. The largest thing in this repo's
# own runs is 235 lines; a megabyte of "kernel" is somebody uploading a blob.
MAX_KERNEL_BYTES = 512 * 1024
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,60}$")


def queue_db() -> sqlite3.Connection:
    conn = sqlite3.connect(QUEUE_DB)
    conn.row_factory = sqlite3.Row
    conn.executescript((HERE / "queue.sql").read_text())
    return conn


def load_tokens() -> dict[str, str]:
    """token -> submitter name. Empty dict disables the write API."""
    if not TOKENS.is_file():
        return {}
    out = {}
    for line in TOKENS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        tok, name = line.split(":", 1)
        if tok.strip():
            out[tok.strip()] = name.strip() or "anonymous"
    return out


def submitter(authorization: str = Header(default="")) -> str:
    tokens = load_tokens()
    if not tokens:
        raise HTTPException(
            503, "submissions are disabled: no token file configured "
                 f"({TOKENS}). See leaderboard/submit.py.")
    prefix = "bearer "
    if not authorization.lower().startswith(prefix):
        raise HTTPException(401, "expected `Authorization: Bearer <token>`")
    name = tokens.get(authorization[len(prefix):].strip())
    if not name:
        raise HTTPException(403, "unknown token")
    return name


class SubmitRequest(BaseModel):
    slug: str = Field(description="Groups jobs into one leaderboard entry. "
                                  "Reusing a slug adds problems to it.")
    problem_key: str = Field(description="e.g. L1__069_rms_norm")
    kernel: str = Field(description="Python source. Must define run().")
    display_name: str | None = None
    model: str | None = None
    notes: str | None = None


class Job(BaseModel):
    id: int
    state: str
    token_name: str
    slug: str
    problem_key: str
    kernel_sha256: str
    kernel_bytes: int
    submitted_utc: str
    started_utc: str | None = None
    finished_utc: str | None = None
    worker: str | None = None
    gpu: int | None = None
    error: str | None = None
    n_workloads: int | None = None
    n_passed: int | None = None
    mean_score: float | None = None
    run_dir: str | None = None
    display_name: str | None = None
    model: str | None = None
    notes: str | None = None
    queue_position: int | None = Field(
        None, description="How many queued jobs are ahead of this one. Null "
                          "once it is no longer queued.")


router = APIRouter(prefix="/api/v1", tags=["submit"])


@router.post("/submit", response_model=Job, status_code=202)
def submit(req: SubmitRequest, who: str = Depends(submitter)):
    """Queue one kernel for one problem. Returns 202 and a job to poll.

    202, not 200: nothing has been measured when this returns. The job is in a
    queue behind however much work is already there, and the GPU it needs runs
    one job at a time.
    """
    if not SLUG_RE.match(req.slug):
        raise HTTPException(422, "slug must be lowercase alphanumeric with "
                                 "'.', '_' or '-', 2-61 chars")
    source = req.kernel
    if not source.strip():
        raise HTTPException(422, "kernel is empty")
    n_bytes = len(source.encode())
    if n_bytes > MAX_KERNEL_BYTES:
        raise HTTPException(413, f"kernel is {n_bytes} bytes; the limit is "
                                 f"{MAX_KERNEL_BYTES}")

    # The problem must exist in the manifest the board is scored against.
    # Checked here so a typo fails in the request rather than on a GPU twenty
    # minutes later, and so the queue can never hold work that cannot be scored.
    from app import db                                    # local: avoids a cycle
    with db() as conn:
        row = conn.execute(
            "SELECT n_scoreable, deferred FROM problem WHERE key=?",
            (req.problem_key,)).fetchone()
    if row is None:
        raise HTTPException(404, f"no such problem: {req.problem_key}")
    if row["deferred"] or not row["n_scoreable"]:
        raise HTTPException(
            409, f"{req.problem_key} has no scoreable workloads in this "
                 f"manifest, so a submission to it cannot be scored. See "
                 f"/methodology#deferred.")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sha = hashlib.sha256(source.encode()).hexdigest()
    spool = spool_path(req.slug, req.problem_key)
    spool.parent.mkdir(parents=True, exist_ok=True)
    spool.write_text(source)

    with queue_db() as conn:
        cur = conn.execute(
            """INSERT INTO job (token_name,slug,display_name,model,problem_key,
                                kernel_sha256,kernel_bytes,notes,submitted_utc)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (who, req.slug, req.display_name, req.model, req.problem_key,
             sha, n_bytes, req.notes, now))
        conn.commit()
        return _job(conn, cur.lastrowid)


@router.get("/jobs", response_model=list[Job])
def jobs(state: str | None = None, slug: str | None = None, limit: int = 100):
    sql = "SELECT * FROM job"
    where, args = [], []
    if state:
        where.append("state=?")
        args.append(state)
    if slug:
        where.append("slug=?")
        args.append(slug)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(max(1, min(limit, 1000)))
    with queue_db() as conn:
        return [_with_position(conn, dict(r)) for r in conn.execute(sql, args)]


@router.get("/jobs/{job_id}", response_model=Job)
def job(job_id: int):
    with queue_db() as conn:
        return _job(conn, job_id)


def _job(conn, job_id: int) -> dict:
    row = conn.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()
    if row is None:
        raise HTTPException(404, f"no such job: {job_id}")
    return _with_position(conn, dict(row))


def _with_position(conn, j: dict) -> dict:
    if j["state"] == "queued":
        j["queue_position"] = conn.execute(
            "SELECT COUNT(*) FROM job WHERE state='queued' AND id < ?",
            (j["id"],)).fetchone()[0]
    else:
        j["queue_position"] = None
    return j


def spool_path(slug: str, problem_key: str) -> Path:
    """Where the submitted source waits for a worker.

    Under the repo, not /tmp: the worker copies it into a sandbox the container
    can see, and a submission that vanishes because someone cleaned /tmp is a
    job that fails for a reason no one can reconstruct.
    """
    return ROOT / "artifacts" / "10" / "_spool" / slug / f"{problem_key}.py"
