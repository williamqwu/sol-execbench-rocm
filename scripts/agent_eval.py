#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate one agent-written kernel — the agent's own feedback loop.

This is what `./evaluate` calls from inside an agent sandbox. It runs the
candidate through the *real* ProblemPackager + eval_driver path, against the
*AMD-derived* tolerances (`artifacts/05/workloads/`), so a kernel that passes
here passes for scoring too. Nothing about the measurement differs between the
agent's loop and the authoritative pass except which GPU it runs on and how
many iterations it times.

**It deliberately never prints T_SOL or T_b.** The agent sees correctness, its
own latency, and the reference latency — the same feedback a kernel engineer
gets from a profiler. Showing the score target would let the agent optimize
against the scoring constants rather than against the hardware, and the
resulting baseline would measure the leak, not the agent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "runners"))

from _common import (  # noqa: E402
    evaluate,
    load_problem,
    problem_key,
    reference_solution,
    summarize,
    write_result,
)


def build_solution(definition, source: str, language: str, entry_point: str):
    """A Solution carrying the agent's source, in whatever language it chose."""
    from sol_execbench.core import Solution

    path = entry_point.split("::")[0]
    return Solution(
        **{
            "name": f"{definition.name}__agent",
            "definition": definition.name,
            "author": "claude-opus-5-agent",
            "spec": {
                "languages": [language],
                "target_hardware": ["LOCAL"],
                "entry_point": entry_point,
                "dependencies": ["torch"],
                "destination_passing_style": False,
            },
            "sources": [{"path": path, "content": source}],
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, type=Path)
    ap.add_argument("--kernel", type=Path)
    # A full Solution JSON, for submissions that cannot be expressed as one
    # bare source file: several sources, or `compile_options` the packager
    # must not have to guess (`-lhipblaslt`, `-lMIOpen`). Same evaluation path
    # as --kernel -- the only difference is where the Solution comes from.
    ap.add_argument("--solution", type=Path,
                    help="path to a Solution JSON (e.g. reference/seeds/*.json)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--language", default="pytorch")
    ap.add_argument("--entry-point", default="kernel.py::run")
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--reference", action="store_true",
                    help="evaluate the problem's own reference instead of --kernel")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    from sol_execbench.core import BenchmarkConfig

    if not (a.reference or a.kernel or a.solution):
        ap.error("one of --kernel, --solution or --reference is required")

    definition, workloads = load_problem(a.problem)
    if a.reference:
        solution = reference_solution(definition)
    elif a.solution:
        from sol_execbench.core import Solution

        sol_dict = json.loads(a.solution.read_text())
        solution = Solution(**sol_dict)
        if solution.definition != definition.name:
            raise SystemExit(
                f"solution targets '{solution.definition}' but --problem is "
                f"'{definition.name}'")
    else:
        solution = build_solution(definition, a.kernel.read_text(),
                                  a.language, a.entry_point)

    config = BenchmarkConfig(warmup_runs=a.warmup, iterations=a.iterations,
                             benchmark_reference=True)

    payload: dict
    try:
        traces = evaluate(definition, workloads, solution, config, timeout=a.timeout)
        payload = summarize(traces)
        payload["ok"] = True
    except BaseException as e:  # noqa: BLE001 - a build/run failure is a result
        import traceback
        payload = {"ok": False, "error": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc(),
                   "workloads": 0, "passed": 0, "all_passed": False,
                   "per_workload": []}

    payload["problem"] = problem_key(a.problem)

    # Speedup over the reference, per workload and in aggregate. This is the
    # agent's objective signal. Aggregate over workloads by the geometric mean:
    # an arithmetic mean of ratios would let one tiny workload with a 10x win
    # mask a regression on the workload that actually costs time.
    ratios = []
    for w in payload.get("per_workload", []):
        lat, ref = w.get("latency_ms"), w.get("reference_latency_ms")
        if w.get("status") == "PASSED" and lat and ref and lat > 0:
            w["speedup_vs_reference"] = ref / lat
            ratios.append(ref / lat)
    if ratios:
        import math
        payload["geomean_speedup"] = math.exp(sum(map(math.log, ratios)) / len(ratios))

    if a.out:
        write_result(a.out, "10-agent-eval", payload)

    if not a.quiet:
        print(render(payload))
    return 0 if payload.get("all_passed") else 1


def render(p: dict) -> str:
    """Compact, agent-readable. No T_SOL, no T_b, no score."""
    lines = []
    if not p.get("ok"):
        lines.append(f"BUILD/RUN FAILED: {p.get('error')}")
        tb = (p.get("traceback") or "").strip().splitlines()
        lines += ["  " + t for t in tb[-25:]]
        return "\n".join(lines)

    n, ok = p.get("workloads", 0), p.get("passed", 0)
    lines.append(f"{ok}/{n} workloads PASSED"
                 + ("" if ok == n else "   <-- correctness must be 100% to count"))
    if p.get("geomean_speedup"):
        lines.append(f"geomean speedup vs reference: {p['geomean_speedup']:.3f}x")
    lines.append("")
    lines.append(f"{'workload':<38} {'status':<8} {'yours(ms)':>11} {'ref(ms)':>10} {'speedup':>8}")
    for w in p.get("per_workload", []):
        uid = (w.get("workload_uuid") or "?")[:36]
        lat, ref = w.get("latency_ms"), w.get("reference_latency_ms")
        sp = w.get("speedup_vs_reference")
        lines.append(
            f"{uid:<38} {str(w.get('status')):<8} "
            f"{(f'{lat:.4f}' if lat else '-'):>11} "
            f"{(f'{ref:.4f}' if ref else '-'):>10} "
            f"{(f'{sp:.2f}x' if sp else '-'):>8}")
        if w.get("status") != "PASSED":
            for ln in (w.get("log") or "").strip().splitlines()[-8:]:
                lines.append(f"    | {ln}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
