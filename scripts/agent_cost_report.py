#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What an agent baseline costs: dollars, wall time, GPUs.

    python scripts/agent_cost_report.py --run artifacts/10/pilot8

Reads `run.json` (sessions, token usage, GPU utilization samples) and
`scored.json` (what the kernels were actually worth), and writes a markdown
report plus a machine-readable summary.

Two things this deliberately does NOT do:

* It does not treat a capped session as a completed one. Sessions killed at
  the spend cap are a **lower bound** on their natural cost, are counted
  separately, and the extrapolation reports a range whose lower end assumes
  capped sessions would have stopped where they were cut off.
* It does not report a single extrapolated number for all 235 problems. Cost
  per problem varies by more than an order of magnitude across the sample, so
  the extrapolation is given as a range from the observed quantiles, with the
  sample size stated next to it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import stamp  # noqa: E402

MANIFEST = ROOT / "artifacts" / "09" / "manifest-v1.json"

# Claude Code's own accounting, inferred exactly from a controlled call:
# 20223 cache-creation + 192 input + 11 output tokens billed at $0.12762875.
# Recorded so the dollar figures can be recomputed if the gateway's contract
# rates differ from list -- the token counts are the measurement, the dollars
# are a conversion.
RATES_USD_PER_MTOK = {"input": 5.0, "output": 25.0,
                      "cache_write": 6.25, "cache_read": 0.50}


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def torch_index_by_card() -> dict[str, int]:
    """`rocm-smi` cardN -> torch/HIP device index, resolved via PCI bus.

    These orderings are scrambled on this node (torch 0 is card3), which
    `scripts/gpu_map.py` documents. `rocm-smi --showuse` reports in card order
    and `HIP_VISIBLE_DEVICES` selects in torch order, so labelling a card's
    busy% with the agent that "had" that index would attribute every agent's
    work to the wrong GPU -- including making the reserved GPU 0 look busy.
    """
    import subprocess
    try:
        out = subprocess.run(
            [str(ROOT / "env" / "solb"), "python", "-c",
             "import sys; sys.path.insert(0,'/work/scripts')\n"
             "from gpu_map import torch_to_amdsmi; import json\n"
             "print(json.dumps(torch_to_amdsmi()))"],
            capture_output=True, text=True, timeout=180)
        m = json.loads(out.stdout.strip().splitlines()[-1])
        return {f"card{v}": int(k) for k, v in m.items()}
    except Exception:      # noqa: BLE001 - report stays useful without it
        return {}


def gpu_utilization(samples: list[dict], card_to_torch: dict[str, int] | None = None) -> dict:
    """Mean and peak busy% per card over the run, and how many were ever busy."""
    if not samples:
        return {}
    cards: dict[str, list[float]] = {}
    for row in samples:
        for k, v in row.items():
            if k == "t":
                continue
            cards.setdefault(k, []).append(float(v))
    c2t = card_to_torch or {}
    per_card = {}
    for c, v in sorted(cards.items()):
        t = c2t.get(c)
        label = f"GPU {t} ({c})" if t is not None else c
        per_card[label] = {"mean": statistics.fmean(v), "max": max(v),
                           "n": len(v), "torch_index": t}
    busy = [c for c, s in per_card.items() if s["max"] > 5.0]
    all_means = [s["mean"] for s in per_card.values()]
    reserved = [c for c, s in per_card.items() if s.get("torch_index") == 0]
    return {
        "per_card": per_card,
        "cards_ever_busy": busy,
        "n_cards_ever_busy": len(busy),
        "node_mean_busy_pct": statistics.fmean(all_means) if all_means else 0.0,
        "samples": len(samples),
        "index_mapping_resolved": bool(c2t),
        "reserved_gpu0_peak_busy_pct":
            max((per_card[c]["max"] for c in reserved), default=None),
        "note": ("Busy% is occupancy, not throughput. An agent spends most of a "
                 "session reading, reasoning and editing; the GPU is idle for "
                 "all of it. This is the number that says how many agents a "
                 "node can actually host."),
    }


def bounds() -> dict:
    m = json.loads(MANIFEST.read_text())
    out = {}
    for key, p in m["problems"].items():
        for uuid, w in p.get("workloads", {}).items():
            if w.get("scoreable") and w.get("t_sol_ms") and w.get("t_b_ms"):
                out[(key, uuid)] = (w["t_sol_ms"], w["t_b_ms"])
    return out


def archive_trajectory(run: dict, dest: Path) -> dict:
    """Copy each sandbox's evaluation record into the run directory.

    The sandboxes live in the container-visible scratch and will be swept.
    Everything needed to reconstruct how a session spent its budget -- the
    per-evaluation results and the snapshot of `kernel.py` that produced each
    one -- is copied next to the artifacts so the analysis outlives the run.
    """
    import shutil
    counts = {}
    for key, s in sorted(run["sessions"].items()):
        sandbox = Path(s.get("sandbox", ""))
        evals = sandbox / "evals"
        if not evals.is_dir():
            continue
        out = dest / key
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(evals.iterdir()):
            if f.is_file():
                shutil.copy2(f, out / f.name)
                n += 1
        # The final submission, and anything it compiles.
        for extra in ("kernel.py", "reference.py", "TASK.md"):
            if (sandbox / extra).is_file():
                shutil.copy2(sandbox / extra, out / extra)
        for f in sandbox.glob("*"):
            if f.is_file() and f.suffix in (".hip", ".cu", ".cpp", ".h", ".py") \
                    and f.name not in ("kernel.py", "reference.py"):
                shutil.copy2(f, out / f.name)
        counts[key] = n
    return counts


def trajectory(run: dict) -> dict:
    """Score every intermediate `./evaluate` the agents ran, in order.

    This answers the question the headline number cannot: sessions saturate
    whatever spend cap they are given, so "cost per problem" is really "the cap
    you chose". What matters is what the marginal dollar buys — and the agents
    already measured that, once per evaluation, for free.

    These evaluations ran on GPUs 1-7 with six other agents loading the node,
    so they are a **trajectory, not a score**. The authoritative numbers are
    the GPU-0 re-times in `scored.json`.
    """
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "_sol_score", ROOT / "src" / "sol_execbench" / "sol_score.py")
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    b = bounds()

    per_problem: dict[str, list] = {}
    for key, s in sorted(run["sessions"].items()):
        sandbox = Path(s.get("sandbox", ""))
        evals = sorted((sandbox / "evals").glob("*.json")) if sandbox else []
        steps = []
        for i, f in enumerate(evals, 1):
            try:
                d = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            scores = []
            for w in d.get("per_workload", []):
                bd = b.get((key, w.get("workload_uuid")))
                if w.get("status") == "PASSED" and bd and w.get("latency_ms"):
                    scores.append(mod.sol_score(w["latency_ms"], bd[1], bd[0]))
            steps.append({
                "n": i,
                "passed": d.get("passed", 0),
                "workloads": d.get("workloads", 0),
                "all_passed": bool(d.get("all_passed")),
                "mean_score": (sum(scores) / len(scores)) if scores else None,
                "geomean_speedup": d.get("geomean_speedup"),
            })
        if steps:
            per_problem[key] = steps

    # Best score reached by evaluation N, averaged over problems that got there.
    by_step: dict[int, list[float]] = {}
    for steps in per_problem.values():
        best = None
        for st in steps:
            if st["all_passed"] and st["mean_score"] is not None:
                best = st["mean_score"] if best is None else max(best, st["mean_score"])
            if best is not None:
                by_step.setdefault(st["n"], []).append(best)

    return {
        "per_problem": per_problem,
        "best_by_eval_index": {
            str(n): {"n_problems": len(v), "mean_best_score": statistics.fmean(v)}
            for n, v in sorted(by_step.items())},
        "_note": ("Measured on GPUs 1-7 under seven-way agent load, so these are "
                  "trajectory values, not scores. The authoritative numbers are "
                  "the idle-GPU-0 re-times in scored.json."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--total-problems", type=int, default=None,
                    help="default: the manifest's scoreable problem count")
    a = ap.parse_args()

    run = json.loads((a.run / "run.json").read_text())
    scored_path = a.run / "scored.json"
    scored = json.loads(scored_path.read_text()) if scored_path.exists() else {}

    total_problems = a.total_problems or json.loads(
        MANIFEST.read_text())["problem_set"]["scoreable_problems"]

    per: list[dict] = []
    for key, s in sorted(run["sessions"].items()):
        sess = s.get("session") or {}
        usage = sess.get("usage") or {}
        sp = (scored.get("per_problem") or {}).get(key, {})
        per.append({
            "problem": key,
            "gpu": s.get("gpu"),
            "cost_usd": sess.get("total_cost_usd") or 0.0,
            "wall_seconds": s.get("wall_seconds") or sess.get("wall_seconds") or 0.0,
            "api_seconds": (sess.get("duration_api_ms") or 0) / 1000.0,
            "turns": sess.get("num_turns"),
            "capped": sess.get("subtype") == "error_max_budget_usd"
                      or sess.get("terminal_reason") == "budget_exhausted",
            "timed_out": bool(sess.get("timed_out")),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_write_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
            "harness_evals": s.get("n_evals"),
            "kernel_changed": s.get("kernel_changed"),
            # scoring outcome
            "workloads": sp.get("workloads"),
            "passed": sp.get("passed"),
            "scored": sp.get("scored"),
            "flagged": sp.get("flagged"),
            "speedup": sp.get("geomean_speedup"),
        })

    costs = [p["cost_usd"] for p in per]
    walls = [p["wall_seconds"] for p in per]
    n_capped = sum(1 for p in per if p["capped"])
    n = len(per)

    tokens = {k: sum(p[f"{k}_tokens"] for p in per)
              for k in ("input", "output", "cache_write", "cache_read")}

    # Per-problem scores, so cost can be read against what it bought.
    by_problem_score: dict[str, float] = {}
    for r in scored.get("results", []):
        if r.get("score") is not None:
            by_problem_score.setdefault(r["problem"], [])
            by_problem_score[r["problem"]].append(r["score"])
    for p in per:
        v = by_problem_score.get(p["problem"])
        p["mean_score"] = statistics.fmean(v) if v else None

    summary = {
        **stamp("10-agent-cost-report"),
        "run_id": run.get("run_id"),
        "model": run.get("model"),
        "gateway": run.get("gateway"),
        "budget_usd_per_session": run.get("budget_usd_per_session"),
        "n_problems": n,
        "n_sessions_capped": n_capped,
        "n_sessions_timed_out": sum(1 for p in per if p["timed_out"]),
        "rates_usd_per_mtok": RATES_USD_PER_MTOK,
        "cost_usd": {
            "total": sum(costs), "mean": statistics.fmean(costs) if costs else 0,
            "median": statistics.median(costs) if costs else 0,
            "min": min(costs) if costs else 0, "max": max(costs) if costs else 0,
            "p25": pct(costs, .25), "p75": pct(costs, .75),
        },
        "wall_seconds": {
            "per_problem_mean": statistics.fmean(walls) if walls else 0,
            "per_problem_median": statistics.median(walls) if walls else 0,
            "per_problem_max": max(walls) if walls else 0,
            "run_total": run.get("wall_seconds_total"),
        },
        "tokens": tokens,
        "gpu": {
            "agents_concurrent_max": len(run.get("gpus_used_by_agents") or []),
            "gpus_available_to_agents": run.get("gpus_used_by_agents"),
            "utilization": gpu_utilization(run.get("gpu_util_samples") or [],
                                           torch_index_by_card()),
        },
        "outcome": scored.get("summary"),
        "per_problem": per,
        "trajectory": trajectory(run),
        "trajectory_archived": archive_trajectory(run, a.run / "trajectory"),
    }
    # Burn rate is the number that generalizes. Cost per problem is set by the
    # cap; dollars per minute of session is a property of the model and the
    # task, and it is what lets someone price a budget they have not tried.
    mins = sum(walls) / 60.0
    summary["burn_rate_usd_per_session_minute"] = (sum(costs) / mins) if mins else 0.0

    # ---- extrapolation to the full benchmark -----------------------------
    #
    # Capped sessions make the mean a lower bound, so both ends are stated.
    concurrency = len(run.get("gpus_available_to_agents") or
                      run.get("gpus_used_by_agents") or [7])
    waves = -(-total_problems // concurrency)          # ceil
    med_cost, mean_cost = summary["cost_usd"]["median"], summary["cost_usd"]["mean"]
    summary["extrapolation_to_full_benchmark"] = {
        "total_problems": total_problems,
        "sample_size": n,
        "concurrency_assumed": concurrency,
        "cost_usd_low": med_cost * total_problems,
        "cost_usd_mid": mean_cost * total_problems,
        "cost_usd_high": pct(costs, .75) * total_problems,
        "wall_hours_at_concurrency":
            waves * summary["wall_seconds"]["per_problem_mean"] / 3600.0,
        "wall_hours_serial":
            total_problems * summary["wall_seconds"]["per_problem_mean"] / 3600.0,
        "caveats": [
            f"{n_capped} of {n} sessions were stopped at the "
            f"${run.get('budget_usd_per_session')} spend cap, so their cost is a "
            f"LOWER bound and so is any figure derived from them.",
            f"n = {n}. Cost per problem spans "
            f"${min(costs) if costs else 0:.2f} to ${max(costs) if costs else 0:.2f} "
            f"in this sample; the range above reflects that spread, not "
            f"confidence in a point estimate.",
            "Dollars are list-price equivalents computed by the Claude Code CLI. "
            "Token counts are the measurement; the AMD gateway bills separately "
            "and its contract rates may differ.",
            "Wall time assumes one agent per GPU. Agents are GPU-idle most of "
            "the time, so oversubscribing is possible -- but it would perturb "
            "the timings each agent optimizes against.",
        ],
    }

    (a.run / "cost-report.json").write_text(json.dumps(summary, indent=1, default=str))
    (a.run / "cost-report.md").write_text(render_md(summary))
    print(render_md(summary))
    print(f"\nwrote {a.run/'cost-report.json'} and {a.run/'cost-report.md'}")
    return 0


def render_md(s: dict) -> str:
    c, w, g = s["cost_usd"], s["wall_seconds"], s["gpu"]
    e = s["extrapolation_to_full_benchmark"]
    u = g.get("utilization") or {}
    L = []
    A = L.append
    A(f"# Agent baseline: what it costs\n")
    A(f"`{s['model']}` via the AMD LLM gateway, driven by the Claude Code CLI, "
      f"on {s['n_problems']} problems sampled across category and headroom.\n")
    A(f"Run `{s['run_id']}` &middot; {s['_provenance']['utc']}\n")

    A("## Headline\n")
    A("| | |")
    A("|---|---|")
    A(f"| cost, {s['n_problems']} problems | **${c['total']:.2f}** |")
    A(f"| cost per problem | median **${c['median']:.2f}**, "
      f"mean ${c['mean']:.2f}, range ${c['min']:.2f}–${c['max']:.2f} |")
    A(f"| wall time per problem | median {w['per_problem_median']/60:.0f} min, "
      f"max {w['per_problem_max']/60:.0f} min |")
    A(f"| wall time, whole run | {(w['run_total'] or 0)/60:.0f} min at "
      f"{g['agents_concurrent_max']} concurrent agents |")
    A(f"| GPUs occupied | {g['agents_concurrent_max']} (one per agent), "
      f"mean busy {u.get('node_mean_busy_pct', 0):.1f}% |")
    if s.get("outcome"):
        o = s["outcome"]
        A(f"| result | {o.get('workloads_scored')} workloads scored, "
          f"mean S = **{o.get('mean_score', 0):.3f}**, "
          f"{o.get('workloads_flagged')} flagged |")
    A("")

    if s["n_sessions_capped"]:
        A(f"> **{s['n_sessions_capped']} of {s['n_problems']} sessions hit the "
          f"${s['budget_usd_per_session']} spend cap** and were stopped mid-work. "
          f"Their cost is what the cap allowed, not what the problem needed, so "
          f"every figure derived from them is a lower bound.\n")

    A("## Per problem\n")
    A("| problem | GPU | cost | wall | turns | evals | workloads passed | mean S | speedup | capped |")
    A("|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|")
    for p in sorted(s["per_problem"], key=lambda r: -r["cost_usd"]):
        passed = (f"{p['passed']}/{p['workloads']}"
                  if p.get("workloads") is not None else "—")
        A(f"| `{p['problem']}` | {p['gpu']} | ${p['cost_usd']:.2f} "
          f"| {p['wall_seconds']/60:.0f} min | {p['turns'] or '—'} "
          f"| {p['harness_evals']} | {passed} "
          f"| {f\"{p['mean_score']:.3f}\" if p.get('mean_score') is not None else '—'} "
          f"| {f\"{p['speedup']:.2f}x\" if p.get('speedup') else '—'} "
          f"| {'yes' if p['capped'] else ''} |")
    A("")

    A("## Tokens\n")
    t = s["tokens"]
    A("| kind | tokens | $/Mtok | cost |")
    A("|---|--:|--:|--:|")
    rates = s["rates_usd_per_mtok"]
    for k, rate_key in (("input", "input"), ("output", "output"),
                        ("cache_write", "cache_write"), ("cache_read", "cache_read")):
        A(f"| {k.replace('_',' ')} | {t[k]:,} | {rates[rate_key]:.2f} "
          f"| ${t[k]*rates[rate_key]/1e6:.2f} |")
    A(f"| **total** | **{sum(t.values()):,}** | | **${c['total']:.2f}** |")
    A("\nToken counts are the measurement. Dollars are the CLI's list-price "
      "conversion; the gateway bills separately.\n")

    A("## GPU concurrency\n")
    A(f"Agents ran on GPUs {g['gpus_available_to_agents']} — "
      f"{g['agents_concurrent_max']} concurrent, one each. GPU 0 was held idle "
      f"and every score was re-measured on it afterwards.\n")
    if u.get("per_card"):
        A("| card | mean busy % | peak busy % |")
        A("|---|--:|--:|")
        for card, st in u["per_card"].items():
            A(f"| {card} | {st['mean']:.1f} | {st['max']:.0f} |")
        A(f"\n{u.get('note','')}\n")

    tr = s.get("trajectory") or {}
    if tr.get("best_by_eval_index"):
        A("## What the marginal dollar buys\n")
        A(f"Every session in this run ended at `budget_exhausted`, which means "
          f"**cost per problem is the cap, not the problem**. The agents burn "
          f"${s.get('burn_rate_usd_per_session_minute', 0):.2f} per minute of session "
          f"and will use whatever they are given. So the useful question is not "
          f"what a problem costs but what more budget buys.\n")
        A("Best mean score reached by the Nth evaluation, over the problems that "
          "got that far:\n")
        A("| evaluations | problems still going | best mean S so far |")
        A("|--:|--:|--:|")
        for n, v in list(tr["best_by_eval_index"].items()):
            A(f"| {n} | {v['n_problems']} | {v['mean_best_score']:.3f} |")
        A(f"\n{tr.get('_note','')}\n")

    A("## Extrapolating to the full benchmark\n")
    A(f"| | |")
    A(f"|---|---|")
    A(f"| problems | {e['total_problems']} scoreable |")
    A(f"| cost | **${e['cost_usd_low']:,.0f} – ${e['cost_usd_high']:,.0f}** "
      f"(median–p75 per problem × {e['total_problems']}) |")
    A(f"| wall time at {e['concurrency_assumed']}-way concurrency | "
      f"**{e['wall_hours_at_concurrency']:.0f} h** |")
    A(f"| wall time serial | {e['wall_hours_serial']:.0f} h |")
    A("")
    A("Caveats, none of them optional:\n")
    for cv in e["caveats"]:
        A(f"* {cv}")
    A("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
