#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The scoring worker. Owns GPU 0, runs one job at a time.

    leaderboard/.venv/bin/python leaderboard/worker.py            # loop
    leaderboard/.venv/bin/python leaderboard/worker.py --once     # one job
    leaderboard/.venv/bin/python leaderboard/worker.py --drain    # until empty

Serialised by design, not by laziness. Every number on this board is timed on
GPU 0 with nothing else on it -- that is what makes T_b, T_SOL and every
submitted score comparable to each other. Two workers, or one worker running
two jobs, would produce numbers that are not comparable to the 3717 already
published. So the throughput ceiling is one submission at a time, and the
right response to a queue backing up is to wait, not to parallelise.

A lock file enforces it across processes: a second worker exits rather than
quietly sharing the GPU.

It does not reimplement scoring. A claimed job is materialised into exactly the
layout `scripts/agent_score.py` already consumes -- a run directory with
`run.json`, sandboxes holding `kernel.py` -- and that script does the re-time,
the reward-hack check, the bound-violation check and the provenance stamp. If
this file computed a score itself, the board would have two scorers that could
disagree, and the disagreement would be invisible.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from submit import queue_db, spool_path      # noqa: E402

SCRATCH = Path(os.environ.get("SOLEXBENCH_SCRATCH", "/var/tmp/solbench"))
LOCK = SCRATCH / "leaderboard-worker.lock"
AGENT_RUNS = ROOT / "artifacts" / "10"
# The ingest's own source configuration. Its presence, not a flag, is what
# tells this worker that ingest knows where the runs live -- see `ingest_cmd()`.
SOURCES = HERE / "sources.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def acquire_lock() -> int:
    """Refuse to start if another worker holds GPU 0.

    O_EXCL, and the pid goes in the file: a stale lock after a crash should be
    diagnosable without guessing. Deliberately not auto-cleared -- a lock left
    by a killed worker may mean a container is still running on the GPU, and
    stealing it would put two jobs on GPU 0, which is the exact thing this
    exists to prevent.
    """
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        holder = LOCK.read_text().strip() if LOCK.exists() else "unknown"
        raise SystemExit(
            f"another worker holds GPU 0 ({holder}).\n"
            f"If you are certain it is dead AND no eval container is still "
            f"running, remove {LOCK} by hand. Do not remove it to 'unstick' a "
            f"long job: a second worker on GPU 0 invalidates both timings.")
    os.write(fd, f"{socket.gethostname()}:{os.getpid()} since {now()}\n".encode())
    return fd


def claim(conn) -> dict | None:
    """Take the oldest queued job, atomically.

    The UPDATE ... WHERE state='queued' is the claim: two workers racing on the
    same row means exactly one of them sees rowcount 1.
    """
    row = conn.execute(
        "SELECT * FROM job WHERE state='queued' ORDER BY id LIMIT 1").fetchone()
    if row is None:
        return None
    cur = conn.execute(
        """UPDATE job SET state='running', started_utc=?, worker=?, gpu=0
            WHERE id=? AND state='queued'""",
        (now(), f"{socket.gethostname()}:{os.getpid()}", row["id"]))
    conn.commit()
    return dict(row) if cur.rowcount == 1 else None


def run_dir_for(slug: str) -> Path:
    return AGENT_RUNS / f"submitted-{slug}"


def materialise(job: dict) -> Path:
    """Lay the job out the way `agent_score.py` expects to find a run.

    Every job for a slug lands in the same run directory, and `run.json` lists
    every problem submitted under it so far. That is what makes a slug a
    leaderboard *entry* rather than one row per kernel: submitting a second
    problem extends the entry instead of creating a rival to it.
    """
    run = run_dir_for(job["slug"])
    run.mkdir(parents=True, exist_ok=True)

    sandbox = SCRATCH / "submitted" / job["slug"] / job["problem_key"]
    sandbox.mkdir(parents=True, exist_ok=True)
    src = spool_path(job["slug"], job["problem_key"])
    if not src.is_file():
        raise FileNotFoundError(f"spooled kernel is gone: {src}")
    (sandbox / "kernel.py").write_text(src.read_text())

    meta = run / "run.json"
    doc = json.loads(meta.read_text()) if meta.exists() else {
        "run_id": f"submitted-{job['slug']}",
        "harness": "leaderboard-submit",
        "sessions": {},
        "note": ("Submitted through the leaderboard API and re-timed on GPU 0 "
                 "by leaderboard/worker.py. The agent-side fields other runs "
                 "carry (turns, cost, trajectory) are absent because there was "
                 "no agent session: a kernel arrived as source."),
    }
    doc["model"] = job.get("model") or doc.get("model")
    doc["display_name"] = job.get("display_name") or doc.get("display_name")
    doc["sessions"][job["problem_key"]] = {
        "problem": job["problem_key"],
        "sandbox": str(sandbox),
        "submitted_by": job["token_name"],
        "submitted_utc": job["submitted_utc"],
        "kernel_sha256": job["kernel_sha256"],
        "job_id": job["id"],
    }
    doc["n_problems"] = len(doc["sessions"])
    meta.write_text(json.dumps(doc, indent=1))
    return run


#: Where each part's scoring manifest lives on this tree. For the error message
#: only -- it is NOT a default; `resolve_manifest` says why.
MANIFEST_BY_PART = {
    "MI355X": "artifacts/09-MI355X/manifest-v4.json",
    "MI350X": "artifacts/09/manifest-v1.2.json",
}


def resolve_manifest(explicit: str | None) -> Path:
    """The manifest this worker scores against -- stated, never guessed.

    `--manifest`, else `$SOLBENCH_MANIFEST`, else a refusal.

    There is deliberately no default. This worker used to invoke
    `agent_score.py` with no manifest at all, and that script defaulted to
    `artifacts/09/manifest-v1.json` -- MI350X's frozen release manifest -- while
    the worker held GPU 0 on an MI355X card. Every submission scored here was
    compared against another part's bounds, worth about +0.08 on mean S, and
    nothing said so.

    A per-part default resolved from the visible cards would fix that particular
    accident and leave the real decision -- WHICH VERSION of this part's manifest
    the board publishes against -- to whichever filename happened to sort last.
    Scores are comparable only within one manifest version, so that decision
    belongs to the operator who also re-ingests the board.
    """
    chosen = explicit or os.environ.get("SOLBENCH_MANIFEST")
    if not chosen:
        raise SystemExit(
            "worker.py must be told which manifest to score against.\n"
            "  Pass --manifest, or set SOLBENCH_MANIFEST. On this tree:\n"
            + "".join(f"    {part}:  {path}\n"
                      for part, path in sorted(MANIFEST_BY_PART.items()))
            + "  It must be THIS node's part. agent_score.py cross-checks the\n"
              "  manifest's part against the node's and refuses, so a wrong\n"
              "  answer here fails loudly rather than publishing.")
    p = Path(chosen)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise SystemExit(f"--manifest {p} does not exist")
    return p


def score(run: Path, timeout: int, manifest: Path) -> subprocess.CompletedProcess:
    """Hand off to the repo's own scorer, on GPU 0.

    `--reuse-retimed` so adding a second problem to a slug does not re-time the
    first: the timing is the expensive part and it has not changed.

    `--manifest` is passed explicitly and is required by `agent_score.py`; see
    `resolve_manifest`.
    """
    cmd = [sys.executable, str(ROOT / "scripts" / "agent_score.py"),
           "--run", str(run), "--gpu", "0", "--reuse-retimed",
           "--manifest", str(manifest)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          cwd=str(ROOT))


def harvest(run: Path, job: dict) -> dict:
    """Pull this job's own numbers out of the run the scorer just wrote."""
    scored = run / "scored.json"
    if not scored.exists():
        return {"error": "scorer produced no scored.json"}
    doc = json.loads(scored.read_text())
    mine = [r for r in doc.get("results", []) if r["problem"] == job["problem_key"]]
    passed = [r for r in mine if r.get("status") == "PASSED"]
    got = [r["score"] for r in passed if r.get("score") is not None]
    return {
        "n_workloads": len(mine),
        "n_passed": len(passed),
        # Over ATTEMPTS, matching the board's `mean (attempted)`: a workload
        # that failed scores zero here rather than leaving the denominator.
        "mean_score": (sum(got) / len(mine)) if mine else None,
        "run_dir": str(run),
        "error": None if mine else "scorer recorded no workloads for this problem",
    }


def served_db() -> Path | None:
    """The database the app will actually serve, asked of the app itself.

    Not re-derived here, and that is the whole point. This file used to spell
    the path `leaderboard/solbench.db`, which `ingest.py` stopped writing when
    the board split one database per part (`db/solbench-<PART>.db`). Nothing
    raised: `extra_roots()` found no roots in a file that was not there and
    rebuilt without them, and the guard in `reingest()` compared an empty set
    against an empty set and reported success. Two independent notions of "the
    database" is how that happened, so there is now one, in `app.py`, and it is
    the same one `run.sh` probes.

    Imported inside the function because importing `app` builds the FastAPI
    application, mounts `static/` and puts `src` on `sys.path` -- side effects
    that claiming and scoring a job has no use for. It is not a way to run
    without the web stack: `submit`, imported at the top of this file, already
    needs fastapi, so worker.py does not start outside `leaderboard/.venv`.
    """
    try:
        import app                                    # leaderboard/.venv only
    except ImportError as exc:
        # Do not fall back to a guessed path -- a second guess is the bug. Say
        # it out loud; `reingest()` turns the resulting None into an error.
        print(f"cannot resolve the served database: {exc} -- worker.py must "
              f"run under leaderboard/.venv", flush=True)
        return None
    try:
        return app.part_databases().get(app.resolve_part())
    except Exception as exc:
        # `resolve_part()` raises on a misconfigured SOLBENCH_PART, where it is
        # an HTTP 500 for a request. Here it would unwind out of `reingest()`
        # after the score was already measured and kill the loop before the job
        # row is updated -- so it becomes None, which `reingest()` reports as a
        # rebuild failure against a job that keeps its score.
        print(f"cannot resolve the served database: {exc}", flush=True)
        return None


def extra_roots() -> list[str]:
    """The `--agent-runs` roots the CURRENT board was built from.

    A bare `ingest.py` reads only `artifacts/10`, so rebuilding without these
    silently deletes every run kept outside the repo. It is not hypothetical:
    the first end-to-end test of this worker scored its job correctly and then
    dropped the 250 USD Opus run off the board, because `reingest()` shelled out
    to a bare `ingest.py`. Same trap as the staleness banner's rebuild command,
    in new code, three commits later.

    Read back out of the database's own `meta` rather than configured here, so
    the roots cannot drift from the ones the last build actually used.
    """
    db = served_db()
    if db is None or not db.is_file():
        return []
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        row = conn.execute(
            "SELECT value FROM meta WHERE key='input_extra_roots'").fetchone()
        conn.close()
        return json.loads(row[0]) if row and row[0] else []
    except Exception:
        return []


def ingest_cmd() -> list[str]:
    """How to shell `ingest.py` so the rebuild keeps everything the board had.

    Two eras, and this worker has to be correct in both:

    * With `leaderboard/sources.json`, ingest reads its own roots and refuses
      to publish a board that lost a submission. Pass nothing: `--agent-runs`
      *overrides* the config rather than adding to it, so handing it the older
      list read out of the last build's `meta` would switch ingest back to the
      stale copy -- the exact drift the config exists to end.
    * Without it, a bare ingest reads only `artifacts/10` and drops every run
      kept outside the repo, so the roots the last build used are replayed.

    Mirrors `ingest.SOURCES` by path rather than importing it, so scoring a job
    does not depend on the ingest module being importable in this process.
    """
    cmd = [str(HERE / ".venv" / "bin" / "python"), str(HERE / "ingest.py")]
    # An operator who pins SOLBENCH_DB has named the file the app serves, so
    # rebuild that one. Unpinned, ingest's own per-part default is canonical and
    # naming it here would just be this file holding an opinion about the path
    # again. `run.sh` makes the same call for the same reason.
    pin = os.environ.get("SOLBENCH_DB")
    if pin:
        cmd += ["--db", pin]
    if SOURCES.is_file():
        return cmd
    roots = extra_roots()
    if roots:
        cmd += ["--agent-runs", *roots]
    return cmd


def reingest() -> str | None:
    """Rebuild the board so the new score is visible. Atomic; never blanks it.

    `ingest.py` now refuses to publish a board that lost a submission, so the
    usual failure arrives as a non-zero exit. The check below is the one it
    cannot make: whether the file it published is the file this app serves.
    """
    before = _current_submissions()
    cmd = ingest_cmd()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       cwd=str(ROOT))
    if p.returncode != 0:
        return (p.stderr or p.stdout)[-2000:]
    # A rebuild that silently loses submissions is worse than one that fails:
    # the board still renders, still looks complete, and the missing entry is
    # only noticed by whoever submitted it.
    after = _current_submissions()
    if after is None:
        # Two causes, and this cannot tell them apart from here: ingest wrote
        # somewhere nothing reads, or the part does not resolve (a bad
        # SOLBENCH_PART). Either way the drop-guard did not run, so the rebuild
        # is not something to report as clean.
        return (f"ingest reported success but no served database is resolvable "
                f"afterwards, so the drop-guard could not run -- ingest wrote "
                f"somewhere nothing reads, or the part does not resolve. "
                f"Command: {' '.join(cmd)}")
    if before is None:
        return None            # nothing to compare against; not a loss
    lost = before - after
    return (f"rebuild dropped {sorted(lost)} from the board -- it did not read "
            f"every agent-run root. Command: {' '.join(cmd)}") if lost else None


def _current_submissions() -> set[str] | None:
    """Slugs on the served board, or None when there is no board to read.

    None rather than an empty set, because the caller subtracts these: with a
    missing or unreadable database `set() - set()` is indistinguishable from
    "nothing was lost", and that is precisely how the drop-guard passed while a
    run fell off the board.
    """
    db = served_db()
    if db is None or not db.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        out = {r[0] for r in conn.execute("SELECT slug FROM submission")}
        conn.close()
        return out
    except Exception:
        return None


def process(job: dict, timeout: int, manifest: Path) -> dict:
    try:
        run = materialise(job)
    except Exception as exc:
        return {"state": "failed", "error": f"{type(exc).__name__}: {exc}"}

    try:
        proc = score(run, timeout, manifest)
    except subprocess.TimeoutExpired:
        # Not "the kernel is wrong". The measurement did not happen, and
        # recording it as a zero score would put an invented number on the
        # board. glm-run1's FlashInfer-Bench__014 is the same failure, found
        # after the fact; here it is recorded at the moment it happens.
        return {"state": "failed",
                "error": f"scoring timed out after {timeout}s -- no measurement"}
    if proc.returncode != 0:
        return {"state": "failed",
                "error": f"agent_score.py exited {proc.returncode}: "
                         f"{(proc.stderr or proc.stdout)[-1500:]}"}

    out = harvest(run, job)
    if out.get("error"):
        return {"state": "failed", **out}
    err = reingest()
    if err:
        # The score is real and on disk; only the board is behind. Saying
        # "failed" would throw away a measurement that did happen.
        out["error"] = f"scored, but the board rebuild failed: {err}"
    return {"state": "scored", **out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="one job, then exit")
    ap.add_argument("--drain", action="store_true",
                    help="run until the queue is empty, then exit")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per job, covering the whole re-time")
    ap.add_argument("--manifest", default=None,
                    help="the scoring manifest, this node's part. Required "
                         "(or $SOLBENCH_MANIFEST); see resolve_manifest().")
    a = ap.parse_args()

    # Resolved BEFORE the lock: refusing after taking GPU 0 would leave a lock
    # file behind for a worker that never ran a job.
    manifest = resolve_manifest(a.manifest)

    fd = acquire_lock()
    print(f"worker up, GPU 0, lock {LOCK}, manifest {manifest.name}",
          flush=True)
    try:
        while True:
            with queue_db() as conn:
                job = claim(conn)
            if job is None:
                if a.once or a.drain:
                    print("queue empty", flush=True)
                    return 0
                time.sleep(a.poll)
                continue

            print(f"[job {job['id']}] {job['slug']} / {job['problem_key']} "
                  f"from {job['token_name']}", flush=True)
            t0 = time.time()
            out = process(job, a.timeout, manifest)
            with queue_db() as conn:
                conn.execute(
                    """UPDATE job SET state=?, error=?, finished_utc=?,
                             n_workloads=?, n_passed=?, mean_score=?, run_dir=?
                        WHERE id=?""",
                    (out["state"], out.get("error"), now(),
                     out.get("n_workloads"), out.get("n_passed"),
                     out.get("mean_score"), out.get("run_dir"), job["id"]))
                conn.commit()
            print(f"[job {job['id']}] {out['state']} in {time.time()-t0:.0f}s"
                  + (f" -- {out['error']}" if out.get("error") else
                     f" -- {out.get('n_passed')}/{out.get('n_workloads')} passed, "
                     f"mean S {out.get('mean_score')}"), flush=True)
            if a.once:
                return 0
    finally:
        os.close(fd)
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
