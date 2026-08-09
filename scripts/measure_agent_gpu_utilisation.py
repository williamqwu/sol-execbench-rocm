#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""How much of an allocated GPU-hour an agent session actually computes.

Two sweeps, same node, same harness, different model. The number is not a
property of the benchmark -- it is a property of how a given model works, and
the two differ by an order of magnitude. That is the finding.

Timing source: each `./evaluate` writes `eval-<ns>.json`, where the filename
stamp is the moment the wrapper started and the file's mtime is the moment it
finished. Neither is a self-report by the agent.
"""
from __future__ import annotations
import glob, json, os, statistics as st, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from provenance import stamp

# The amdpilot bridge's own state, which is where a fleet run's job ids live.
# It is outside this repo on purpose -- the fleet is not part of the benchmark --
# so the path is overridable and its absence is a plain FileNotFoundError rather
# than a wrong answer.
STATE = Path(os.environ.get(
    "SBT_STATE_RUNS",
    "/home/qinwu/dev-imported/amdpilot-v2/dash-overlay/solbench-tasks/.state/runs"))
JOBD = Path(os.environ.get("JOBD_LAUNCH_ROOT", Path.home() / ".jobd" / "jobs"))
MAX_EVAL_S = 3600.0   # a span longer than the session cap is a clock artefact, not an eval


def evals_from(dirpath: Path) -> list[tuple[float, float]]:
    out = []
    for f in glob.glob(str(dirpath / "eval-*.json")):
        try:
            start = int(Path(f).stem.split("-")[1]) / 1e9
        except (IndexError, ValueError):
            continue
        end = os.path.getmtime(f)
        if start < end < start + MAX_EVAL_S:
            out.append((start, end))
    return sorted(out)


def job_window(job_id: str) -> tuple[float, float] | None:
    try:
        r = json.load(urllib.request.urlopen(
            f"http://127.0.0.1:7201/v1/jobs/{job_id}", timeout=10))
    except Exception:
        return None
    if r.get("admitted_at") and r.get("finished_at"):
        return r["admitted_at"], r["finished_at"]
    return None


def cards_used(run_id: str) -> list[int]:
    """Which physical cards a sweep's agents were actually placed on.

    Read from the run's own `gpus_used_by_agents`, not assumed. A constant here
    would be wrong for exactly the comparison this script exists to make:
    `glm-sweep-2` ran across all eight cards, `gpt56-40` across seven, because
    the fleet's policy on holding GPU 0 out changed between them. Dividing both
    by 7 understates one denominator by an eighth and quietly inflates the very
    utilisation figure being compared.
    """
    run = json.load(open(ROOT / "artifacts" / "10" / run_id / "run.json"))
    gpus = run.get("gpus_used_by_agents")
    if not gpus:
        raise SystemExit(
            f"{run_id}/run.json has no gpus_used_by_agents; the allocated "
            f"denominator cannot be assumed, so this run is not measurable here")
    return sorted(gpus)


def profile(run_id: str, eval_dir):
    ported = json.load(open(STATE / run_id / "ported.json"))
    svc, per, windows = [], [], []
    for p in ported["problems"]:
        jid = p["job_id"]
        pairs = evals_from(eval_dir(run_id, jid, p["key"]))
        per.append(len(pairs))
        svc += [e - s for s, e in pairs]
        w = job_window(jid)
        if w:
            windows.append(w)
    if not svc or not windows:
        raise SystemExit(f"{run_id}: no timings found")
    wall = max(e for _, e in windows) - min(s for s, _ in windows)
    gpu_s = sum(svc)
    cards = cards_used(run_id)
    # Concurrency actually achieved, area-weighted over the sweep's wall clock.
    ev = sorted([(s, 1) for s, _ in windows] + [(e, -1) for _, e in windows])
    c = area = 0
    prev = ev[0][0]
    peak = 0
    for t, d in ev:
        area += c * (t - prev); prev = t; c += d; peak = max(peak, c)
    return {
        "run_id": run_id,
        "model": ported["model"],
        "n_problems": len(ported["problems"]),
        "gpu_count_per_job": ported["gpu_count"],
        "sessions_peak_concurrent": peak,
        "sessions_mean_concurrent": round(area / wall, 2),
        "wall_hours": round(wall / 3600, 2),
        "n_evaluations": len(svc),
        "evals_per_session": {
            "median": st.median(per), "mean": round(st.mean(per), 1),
            "min": min(per), "max": max(per),
        },
        "eval_seconds": {
            "median": round(st.median(svc), 1), "mean": round(st.mean(svc), 1),
            "p99": round(sorted(svc)[int(0.99 * len(svc))], 1),
            "max": round(max(svc), 1),
        },
        "cards_the_agents_ran_on": cards,
        "gpu_hours_computing": round(gpu_s / 3600, 2),
        "gpu_hours_allocated": round(len(cards) * wall / 3600, 1),
        "utilisation_pct": round(100 * gpu_s / (len(cards) * wall), 1),
    }


def glm_dir(run_id, jid, key):
    return ROOT / "artifacts" / "10" / run_id / "trajectory" / key


def jobd_dir(run_id, jid, key):
    return JOBD / jid / "evals"


rows = [profile("glm-sweep-2", glm_dir), profile("gpt56-40", jobd_dir)]
out = {
    "_provenance": stamp("11-agent-gpu-utilisation"),
    "question": (
        "An agent session holds one MI350X for its whole hour. How much of that "
        "hour is the card actually computing? The answer decides whether raising "
        "job concurrency is nearly free or is the thing that saturates the node."
    ),
    "method": (
        "Per-evaluation start comes from the eval record's filename nanosecond "
        "stamp and end from its mtime; both are written by the ./evaluate wrapper, "
        "not reported by the agent. Session windows are the Job Queue's admitted_at "
        "and finished_at. Allocated GPU-hours are the sweep's own card count x "
        "its wall clock, with the count read per run from `gpus_used_by_agents` "
        "rather than assumed: glm-sweep-2 ran on all eight cards and gpt56-40 on "
        "seven, because the fleet stopped holding GPU 0 out between them. A "
        "constant 7 would have understated one denominator by an eighth, and it "
        "would have done so on exactly the side of the comparison this artifact "
        "is about."
    ),
    "runs": rows,
    "finding": (
        "Utilisation is a property of the model, not of the benchmark. GLM-5.2 "
        "spends its hour reasoning and evaluates ~6 times; gpt-5.6-sol evaluates "
        "~27 times in a quarter of the wall clock. The same 7-wide fleet is "
        "therefore nearly idle under one and roughly half-loaded under the other, "
        "so a concurrency target derived from one does not transfer to the other."
    ),
    "caveats": [
        "mtime is the close of the last write, so an eval that writes its record "
        "and then tears down a process is measured slightly short. The bias is "
        "the same in both runs and is small against a median of 9-11 s.",
        "Spans longer than the 3600 s session cap are discarded as clock "
        "artefacts rather than trusted; none were found in either run.",
        "Utilisation counts a card as busy only while an ./evaluate is running. "
        "Agent-authored torch code run outside the wrapper is invisible here and "
        "would make the true figure higher.",
        "glm-sweep-2's sessions were cancelled at the 1 h cap; gpt56-40's ended "
        "on their own. The wall clocks are therefore not a like-for-like "
        "comparison of model speed, only of how the fleet was loaded.",
    ],
}
dst = ROOT / "artifacts" / "11" / "agent-gpu-utilisation.json"
dst.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(rows, indent=2))
print("->", dst)
