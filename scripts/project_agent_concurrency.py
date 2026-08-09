#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What raising job concurrency would buy, simulated on measured distributions.

This is a projection and not a measurement. Every input is empirical -- the
per-evaluation service times and the think gaps between them, taken from a real
sweep -- but the output is a simulation of a fleet configuration that has never
been run. Nothing here is a benchmark result.
"""
from __future__ import annotations
import glob, heapq, json, os, random, statistics as st, sys, urllib.request
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
CARDS = 7
TARGET_PROBLEMS = 220
SEEDS = 20


def sessions_for(run_id: str) -> list[dict]:
    """Each session as an alternating (think, evaluate) trace."""
    ported = json.load(open(STATE / run_id / "ported.json"))
    out = []
    for p in ported["problems"]:
        jid = p["job_id"]
        pairs = []
        for f in glob.glob(str(JOBD / jid / "evals" / "eval-*.json")):
            s = int(Path(f).stem.split("-")[1]) / 1e9
            e = os.path.getmtime(f)
            if s < e < s + 3600:
                pairs.append((s, e))
        pairs.sort()
        try:
            r = json.load(urllib.request.urlopen(
                f"http://127.0.0.1:7201/v1/jobs/{jid}", timeout=10))
        except Exception:
            continue
        t0, t1 = r.get("admitted_at"), r.get("finished_at")
        if not (t0 and t1 and pairs):
            continue
        gaps, prev = [], t0
        for s, e in pairs:
            gaps.append(max(0.0, s - prev)); prev = e
        gaps.append(max(0.0, t1 - prev))
        out.append({"svc": [e - s for s, e in pairs], "gaps": gaps})
    return out


def simulate(pool, n_concurrent, n_cards, n_problems, seed):
    """FIFO card broker, `n_concurrent` sessions in flight, `n_cards` cards.

    A session alternates think -> ask for a card -> evaluate -> release. A card
    is held only for the evaluate, which is exactly what the lease broker does;
    the seven-wide fleet today holds one for the whole session instead.
    """
    rng = random.Random(seed)
    jobs = [dict(pool[rng.randrange(len(pool))]) for _ in range(n_problems)]
    for j in jobs:
        j["i"] = 0
    pending = list(range(n_problems))
    ev, seq, busy, waitq, waits, last = [], 0, 0, [], [], 0.0

    def push(t, kind, idx):
        nonlocal seq
        seq += 1
        heapq.heappush(ev, (t, seq, kind, idx))

    for _ in range(min(n_concurrent, n_problems)):
        i = pending.pop(0)
        push(jobs[i]["gaps"][0], "want", i)

    def grant(t, idx):
        nonlocal busy
        busy += 1
        push(t + jobs[idx]["svc"][jobs[idx]["i"]], "release", idx)

    while ev:
        t, _, kind, idx = heapq.heappop(ev)
        if kind == "want":
            if busy < n_cards:
                waits.append(0.0); grant(t, idx)
            else:
                waitq.append((t, idx))
        elif kind == "release":
            busy -= 1
            j = jobs[idx]; j["i"] += 1
            if waitq:
                qt, qi = waitq.pop(0)
                waits.append(t - qt); grant(t, qi)
            gap = j["gaps"][j["i"]]
            if j["i"] >= len(j["svc"]):
                push(t + gap, "end", idx)
            else:
                push(t + gap, "want", idx)
        else:
            last = max(last, t)
            if pending:
                i = pending.pop(0)
                push(t + jobs[i]["gaps"][0], "want", i)
    waits.sort()
    return last, waits


pool = sessions_for("gpt56-40")
gpu_s_per_session = st.mean([sum(s["svc"]) for s in pool])
rows = []
for n in (7, 10, 14, 20, 28, 40, 60):
    walls, p50, p95, p99 = [], [], [], []
    for seed in range(SEEDS):
        w, waits = simulate(pool, n, CARDS, TARGET_PROBLEMS, seed)
        walls.append(w)
        p50.append(waits[len(waits) // 2])
        p95.append(waits[int(0.95 * len(waits))])
        p99.append(waits[int(0.99 * len(waits))])
    wall = st.mean(walls)
    rows.append({
        "sessions_concurrent": n,
        "wall_hours": round(wall / 3600, 1),
        "gpu_utilisation_pct": round(
            100 * gpu_s_per_session * TARGET_PROBLEMS / (CARDS * wall), 0),
        "card_wait_p50_s": round(st.mean(p50), 1),
        "card_wait_p95_s": round(st.mean(p95), 1),
        "card_wait_p99_s": round(st.mean(p99), 1),
        "wait_share_of_budget_pct": round(
            100 * st.mean(p50) * st.mean([len(s["svc"]) for s in pool]) / 3600, 1),
    })

out = {
    "_provenance": stamp("11-concurrency-projection"),
    "kind": "simulation",
    "warning": (
        "Not a measurement. Service and think times are empirical; the fleet "
        "configurations below have never been run. Quote as a projection only."
    ),
    "question": (
        "gpt-5.6-sol already keeps the 7-card fleet 46% busy at 7 sessions wide. "
        "How much wall clock is left to recover by decoupling the card from the "
        "session -- and at what concurrency does the queue wait start eating the "
        "agent's own 3600 s budget?"
    ),
    "model_of_the_fleet": (
        "One FIFO broker over 7 cards. A session holds a card only for the "
        "duration of one ./evaluate, matching scripts/gpu_broker.py. Sessions are "
        "bootstrapped with replacement from the 40 real gpt56-40 traces; a "
        "session's think gaps are taken as independent of how long it waited, "
        "which is optimistic for an agent that would otherwise be reading output."
    ),
    "inputs": {
        "source_run": "gpt56-40",
        "n_source_sessions": len(pool),
        "cards": CARDS,
        "problems_simulated": TARGET_PROBLEMS,
        "seeds_per_row": SEEDS,
        "mean_gpu_seconds_per_session": round(gpu_s_per_session, 0),
        "mean_evals_per_session": round(
            st.mean([len(s["svc"]) for s in pool]), 1),
    },
    "rows": rows,
    "reading": (
        "The floor is set by total GPU demand, not by concurrency: 220 problems "
        "need about 33 GPU-hours, and 7 cards cannot deliver that in under ~4.7 h "
        "however many sessions are in flight. Concurrency past ~20 buys no wall "
        "clock and only adds queue wait -- which is charged to the agent's 3600 s "
        "session budget, so it is a correctness cost, not just a latency one. The "
        "useful band is 14-20."
    ),
    "does_not_model": [
        "The Model API front door's in-flight cap, which is 16 today and is the "
        "other candidate bottleneck; these rows assume the API is never the "
        "constraint.",
        "Contention between concurrent leases. Measured separately at 14 clients "
        "over 7 cards and found within +-2.5% (artifacts/11/gpu-broker-validation.json), "
        "but not re-measured at 40.",
        "An agent that changes its behaviour when evaluations get slower -- it may "
        "evaluate less, which would lower demand, or retry more, which would raise it.",
    ],
}
dst = ROOT / "artifacts" / "11" / "concurrency-projection.json"
dst.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(rows, indent=2))
print("->", dst)
