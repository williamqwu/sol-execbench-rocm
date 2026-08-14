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


def _settle_digest(per_workload: list[dict]) -> dict:
    """Per-arm summary of the pre-window settle, from the traces themselves.

    Empty-ish under the locked basis, where the driver neither brackets nor
    settles and `clock_bracket` is None -- which is itself the answer, so the
    field is always present rather than omitted when off.
    """
    br = [w.get("clock_bracket") or {} for w in per_workload]
    br = [b for b in br if b]
    if not br:
        return {"enabled": False, "n_with_bracket": 0}
    # NESTED, not flat. `ClockBracket.settle` is the settle record or None; the
    # flat keys on the bracket are the clock samples. Reading `settle_enabled`
    # off the top level finds nothing and reports every arm as unsettled, which
    # is the wrong answer in the direction that looks like a finding.
    on = [b["settle"] for b in br if isinstance(b.get("settle"), dict)]
    settled = [s for s in on if s.get("settled")]
    capped = [s for s in on if s.get("settle_capped")]
    refused = [b for b in br if b.get("clock_bracket_refused")]
    ms = [s.get("settle_ms") for s in on if s.get("settle_ms") is not None]
    return {
        "enabled": bool(on),
        "n_with_bracket": len(br),
        "n_settle_enabled": len(on),
        "n_settled": len(settled),
        "n_settle_capped": len(capped),
        "n_bracket_refused": len(refused),
        "median_settle_ms": (sorted(ms)[len(ms) // 2] if ms else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--order", default="hip_events,rocprof",
                    help="comma-separated arm order. The DEFAULT is the "
                         "original order and every existing artifact was taken "
                         "under it; the flag exists so the order can be "
                         "TESTED, not so it can be varied casually. Under an "
                         "unlocked clock basis the second arm runs on a warmer "
                         "card, so a divergence that flips sign when this is "
                         "reversed is thermal and not a property of either "
                         "methodology.")
    a = ap.parse_args()

    order = tuple(x.strip() for x in a.order.split(",") if x.strip())
    if sorted(order) != ["hip_events", "rocprof"]:
        ap.error(f"--order must name both arms exactly once, got {order!r}")

    def body() -> dict:
        import os

        from sol_execbench.core import BenchmarkConfig

        definition, workloads = load_problem(a.problem)
        solution = reference_solution(definition)
        config = BenchmarkConfig(warmup_runs=a.warmup, iterations=a.iterations,
                                 benchmark_reference=False)

        per_method: dict[str, dict] = {}
        for methodology in order:
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
                    # Whether the card was settled before each window OPENED,
                    # per arm. A comparison of two methodologies is only a
                    # comparison of the methodologies if both arms met the
                    # window in the same thermal state; recorded so that can be
                    # checked from the artifact instead of assumed from the
                    # command line that produced it.
                    "settle": _settle_digest(summary["per_workload"]),
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
            # Recorded because it is a confound, not a setting: under an
            # unlocked clock basis the arm that runs second sees a warmer card.
            # An artifact that does not say which arm went first cannot be
            # compared against one taken the other way round.
            "arm_order": list(order),
            "per_method": per_method,
            "divergences": divergences,
            "n_compared": len(divergences),
        }

    return run_guarded(a.out, "04-methodology-compare", body)


if __name__ == "__main__":
    raise SystemExit(main())
