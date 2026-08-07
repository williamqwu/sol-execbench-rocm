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


def fleet_spec_cap(detail: dict[str, dict], job_ids: set[str]) -> int | None:
    """The wall-clock cap the fleet placed THIS RUN's jobs under, from its spec.

    Read, not assumed. Every J2 job carries `spec.eta_s` and the daemon SIGTERMs
    at it, so the cap is a recorded fact about the run and belongs on the board
    the way pilot8's dollar budget does: a session that was stopped did not
    choose when to stop, and that changes how its score reads.

    Scoped to this run's job ids, because the fleet database holds every J2 job
    ever placed and they are not all under the same cap -- the first row in the
    table this was written against says 1800 s on an MI355X. Taking the cap
    from the whole table answers a question about the fleet and labels it as a
    fact about the run. If this run's own jobs disagree there is no single cap
    to state, and this returns None rather than picking one of them.
    """
    caps = {d["_eta_s"] for j, d in detail.items()
            if j in job_ids and d.get("_eta_s")}
    return int(caps.pop()) if len(caps) == 1 else None


def trajectory_series(run: dict) -> dict:
    """The per-evaluation score series, from the benchmark's own scorer.

    Imported rather than reimplemented: `agent_cost_report.trajectory()` is
    what built pilot8's series, it resolves bounds out of the frozen manifest,
    and it calls `sol_score` from `src/`. A second copy of that arithmetic here
    would be a second answer to the same question, drifting quietly.
    """
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "_agent_cost_report", ROOT / "scripts" / "agent_cost_report.py")
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.trajectory(run)


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
        for jid, state, dj, sj in conn.execute(
                "SELECT id, state, detail_json, spec_json FROM jobs_j2"):
            try:
                out[jid] = {"state": state, **(json.loads(dj or "{}"))}
            except json.JSONDecodeError:
                out[jid] = {"state": state}
            try:
                out[jid]["_eta_s"] = (json.loads(sj or "{}")).get("eta_s")
            except json.JSONDecodeError:
                pass
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
    cap = fleet_spec_cap(detail, {s.get("job_id") for s in run["sessions"].values()})

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
            # And into the staged sandbox, which `sbt collect` fills with
            # kernel.py and reference.py only. `agent_cost_report.trajectory()`
            # reads the evals from there, and it is the scorer pilot8's series
            # came from -- pointing it at these rather than reimplementing the
            # per-eval score here keeps one formula in the tree.
            staged = Path(sess.get("sandbox", "")) / "evals"
            if not a.dry_run:
                dest.mkdir(parents=True, exist_ok=True)
                staged.mkdir(parents=True, exist_ok=True)
            for f in evals:
                stamp = f.stem.split("-", 1)[1]
                snap = sandbox / "evals" / f"kernel-{stamp}.py"
                if not a.dry_run:
                    shutil.copy2(f, dest / f.name)
                    shutil.copy2(f, staged / f.name)
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
            # `capped` is the fleet's wall clock, not a budget in dollars.
            # Recorded because a run that was stopped is not a run that
            # finished, and every downstream reading of its score depends on
            # knowing which.
            "capped": bool(cap and wall and wall >= cap - 5),
            "timed_out": d.get("state") == "cancelled",
            "gpu": (d.get("gpus") or [None])[0],
        })

    report = {
        "_note": ("Effort as the fleet recorded it. cost_usd is null on every "
                  "problem: the model gateway returns 0.0 for models it has no "
                  "price for, and a zero here would be a cost nobody measured. "
                  "Token counts are real."),
        # The constraint this run was under, in the units it was actually
        # imposed in. `ingest.py` labels the trial from it, the way it labels
        # pilot8 from `budget_usd_per_session`.
        "wall_cap_seconds": cap,
        "per_problem": per_problem,
        # The per-evaluation series the run page plots, scored by
        # `agent_cost_report.trajectory()` -- the same function, against the
        # same manifest bounds, that produced pilot8's. Its own docstring is
        # the caveat that matters: these evaluations ran on GPUs 1-7 with the
        # rest of the fleet loading the node, so the series is a TRAJECTORY,
        # not a score. The authoritative numbers are the GPU-0 re-times in
        # scored.json, and the page labels them differently for that reason.
        "trajectory": trajectory_series(run) if not a.dry_run else {},
    }
    if not a.dry_run:
        (a.run / "cost-report.json").write_text(json.dumps(report, indent=1))

    print(f"{'would import' if a.dry_run else 'imported'}: "
          f"{n_traj} trajectories ({n_snap} kernel snapshots), "
          f"{n_tx} transcripts, {len(per_problem)} effort rows")
    capped = sum(p["capped"] for p in per_problem)
    print(f"  fleet wall cap: {cap if cap else 'not recorded'}"
          f"{'s' if cap else ''} -- {capped}/{len(per_problem)} reached it")
    if missing:
        print(f"  {len(missing)} sessions had no sandbox on this host: "
              f"{missing[:4]}{' ...' if len(missing) > 4 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
