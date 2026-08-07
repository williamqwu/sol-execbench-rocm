#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Move a fleet run's trajectory, effort and transcripts into the run layout.

`sbt collect` (dash-overlay) writes `run.json` and stages `kernel.py` for the
scorer, and stops there — correctly, since the score is the benchmark's to
compute. But three other things the fleet already recorded never make the
crossing, and without them a run lands on the board as a column of final
numbers with no account of how they were reached:

    trajectory/<key>/eval-<ts>.json     every ./evaluate the agent ran
    trajectory/<key>/kernel-<ts>.py     the source at that moment
    transcripts/<key>.jsonl             what the agent actually did
    cost-report.json                    wall time, tokens, evals, per problem

All of it exists in the job sandbox under `~/.jobd/jobs/<job-id>/`. This script
copies it into the layout `leaderboard/ingest.py` already reads. It computes
nothing and re-times nothing.

**Cost is written as null, never zero.** The fleet's gateway returns
`total_cost_usd: 0.0` for models it has no price for, and GLM-5.2 is one — 420
million tokens did not cost nothing, and a zero on the board would be a
measurement nobody made. Tokens are recorded because they were.

Usage:
    python3 scripts/import_fleet_depth.py --run artifacts/10/<run-id>
    python3 scripts/import_fleet_depth.py --run artifacts/10/<run-id> --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = Path.home() / ".jobd" / "jobs"
FLEET_DB = (Path.home() / "dev-imported" / "amdpilot-v2" / "dash-overlay"
            / "database" / ".state" / "db.sqlite3")

# The J2 wall-clock cap. A job that reaches it was killed by SIGTERM mid-work,
# which is a different ending from an agent that decided it was done, and the
# board has a column for saying so.
CAP_SECONDS = 3600


def fleet_detail() -> dict[str, dict]:
    """`{job_id: detail}` from the fleet Database, or `{}` if it is not here.

    Read-only and optional: the sandboxes carry the trajectory on their own, so
    a missing fleet database costs the effort columns, not the run.
    """
    if not FLEET_DB.exists():
        print(f"note: no fleet database at {FLEET_DB}; "
              f"wall time and job state will be absent")
        return {}
    out = {}
    conn = sqlite3.connect(f"file:{FLEET_DB}?mode=ro", uri=True)
    try:
        for jid, state, dj in conn.execute(
                "SELECT id, state, detail_json FROM jobs_j2"):
            try:
                out[jid] = {"state": state, **(json.loads(dj or "{}"))}
            except json.JSONDecodeError:
                out[jid] = {"state": state}
    finally:
        conn.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=lambda p: Path(p).resolve(),
                    help="artifacts/10/<run-id> (must contain run.json)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    run = json.loads((a.run / "run.json").read_text())
    detail = fleet_detail()

    n_traj = n_snap = n_tx = 0
    per_problem: list[dict] = []
    missing: list[str] = []

    for key, sess in sorted(run["sessions"].items()):
        job = sess.get("job_id")
        sandbox = JOBS / job if job else None
        if not sandbox or not sandbox.is_dir():
            missing.append(key)
            continue
        d = detail.get(job, {})

        evals = sorted((sandbox / "evals").glob("eval-*.json"))
        if evals:
            dest = a.run / "trajectory" / key
            if not a.dry_run:
                dest.mkdir(parents=True, exist_ok=True)
            for f in evals:
                stamp = f.stem.split("-", 1)[1]
                snap = sandbox / "evals" / f"kernel-{stamp}.py"
                if not a.dry_run:
                    shutil.copy2(f, dest / f.name)
                    if snap.exists():
                        shutil.copy2(snap, dest / snap.name)
                n_snap += int(snap.exists())
            n_traj += 1

        # Verbatim. The counting side of `ingest.py` knows two transcript
        # shapes; rewriting an agent's own record into the other one would put
        # a translation on the page under the word "transcript".
        events = sandbox / "codex-events.jsonl"
        if events.exists():
            if not a.dry_run:
                (a.run / "transcripts").mkdir(parents=True, exist_ok=True)
                shutil.copy2(events, a.run / "transcripts" / f"{key}.jsonl")
            n_tx += 1

        usage = (sess.get("session") or {}).get("usage") or {}
        wall = d.get("wall_s") or d.get("elapsed_s")
        per_problem.append({
            "problem": key,
            # See the module docstring: the gateway prices this model at zero,
            # which is not the same as it having been free.
            "cost_usd": None,
            "wall_seconds": wall,
            "api_seconds": None,
            "turns": d.get("turns") or usage.get("calls"),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "cache_write_tokens": None,
            "cache_read_tokens": None,
            "harness_evals": len(evals),
            "kernel_changed": sess.get("kernel_changed"),
            # `capped` is the fleet's hour, not a budget in dollars. Recorded
            # because a run that was stopped is not a run that finished, and
            # every downstream reading of its score depends on knowing which.
            "capped": bool(wall and wall >= CAP_SECONDS - 5),
            "timed_out": d.get("state") == "cancelled",
            "gpu": (d.get("gpus") or [None])[0],
        })

    report = {
        "_note": ("Effort as the fleet recorded it. cost_usd is null on every "
                  "problem: the model gateway returns 0.0 for models it has no "
                  "price for, and a zero here would be a cost nobody measured. "
                  "Token counts are real."),
        "per_problem": per_problem,
    }
    if not a.dry_run:
        (a.run / "cost-report.json").write_text(json.dumps(report, indent=1))

    print(f"{'would import' if a.dry_run else 'imported'}: "
          f"{n_traj} trajectories ({n_snap} kernel snapshots), "
          f"{n_tx} transcripts, {len(per_problem)} effort rows")
    capped = sum(p["capped"] for p in per_problem)
    print(f"  {capped}/{len(per_problem)} hit the {CAP_SECONDS}s fleet cap")
    if missing:
        print(f"  {len(missing)} sessions had no sandbox on this host: "
              f"{missing[:4]}{' ...' if len(missing) > 4 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
