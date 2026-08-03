#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 04 runner — hip_events vs rocprof divergence for one problem.

Both methodologies time the SAME solution on the SAME inputs back to back in
one process, so the comparison is of the two measurement methods and not of
two moments in the node's life.

The expected result is that `rocprof` reads LOWER, because event pairs bracket
the host launch and dispatch-level activity tracing does not. On a 4096-cube
BF16 GEMM the gap is about 2.9%; on microsecond kernels it should be much
larger, since a fixed launch overhead is a bigger fraction of a smaller
number. That is the finding, not an error — which is why the acceptance check
reports μs-scale kernels separately instead of folding them into one median.

    python scripts/runners/compare_methodology.py --problem <dir> --out <file>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    PASSED,
    evaluate,
    load_problem,
    problem_key,
    reference_solution,
    run_guarded,
    summarize,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()

    def body() -> dict:
        import os

        from sol_execbench.core import BenchmarkConfig

        definition, workloads = load_problem(a.problem)
        solution = reference_solution(definition)
        config = BenchmarkConfig(warmup_runs=a.warmup, iterations=a.iterations,
                                 benchmark_reference=False)

        per_method: dict[str, dict] = {}
        for methodology in ("hip_events", "rocprof"):
            # The eval driver resolves the methodology from the vendor
            # default, so it is selected by environment here rather than by
            # argument -- and the driver records what it used on every trace,
            # which is what the comparison below reads back.
            os.environ["SOLEXBENCH_METHODOLOGY"] = methodology
            try:
                traces = evaluate(definition, workloads, solution, config,
                                  timeout=a.timeout)
                summary = summarize(traces)
                per_method[methodology] = {
                    "ok": True,
                    "recorded_methodology": sorted(
                        {w["methodology"] for w in summary["per_workload"]}
                    ),
                    "latency_ms": {
                        w["workload_uuid"]: w["latency_ms"]
                        for w in summary["per_workload"]
                        if w["status"] == PASSED and w["latency_ms"]
                    },
                }
            except Exception as e:                    # noqa: BLE001
                per_method[methodology] = {"ok": False,
                                           "error": f"{type(e).__name__}: {e}"}
        os.environ.pop("SOLEXBENCH_METHODOLOGY", None)

        ev = (per_method.get("hip_events") or {}).get("latency_ms") or {}
        rp = (per_method.get("rocprof") or {}).get("latency_ms") or {}
        divergences = []
        for uuid in sorted(set(ev) & set(rp)):
            e, r = ev[uuid], rp[uuid]
            if e > 0:
                divergences.append({
                    "workload_uuid": uuid,
                    "hip_events_ms": e,
                    "rocprof_ms": r,
                    # Signed: the direction is the finding. A positive value
                    # means events read slower, which is the expected sign.
                    "divergence_pct": 100.0 * (e - r) / e,
                    # Sub-100 us kernels are where launch overhead dominates,
                    # and they are reported separately rather than averaged in.
                    "microsecond_scale": e < 0.1,
                })
        return {
            "problem": problem_key(a.problem),
            "definition": definition.name,
            "per_method": per_method,
            "divergences": divergences,
            "n_compared": len(divergences),
        }

    return run_guarded(a.out, "04-methodology-compare", body)


if __name__ == "__main__":
    raise SystemExit(main())
