#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 10 — run coding agents across the benchmark.

    # pilot: 6 problems per category, both harnesses
    python scripts/run_agents.py --run-id pilot-01 --limit-per-category 6

    # everything
    python scripts/run_agents.py --run-id full-01

    # one harness, one category, for debugging the loop
    python scripts/run_agents.py --run-id smoke --harness claude-code \\
        --categories L1 --limit-per-category 1

Writes one directory per (harness, problem) under
``artifacts/10/runs/<run-id>/``, each holding the agent's packet and a
``session.json``. Resumable: re-running skips units that already have a
session. Scoring is a separate step -- ``scripts/score_solutions.py`` -- because
nothing an agent produced is trusted to score itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from provenance import stamp  # noqa: E402
from solexbench_agents import (  # noqa: E402
    HARNESSES,
    Sweep,
    Unit,
    default_agent_gpus,
    discover_problems,
    load_deferred,
    preflight,
)
from solexbench_agents.scoring import tree_digest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-id", required=True,
                    help="names the output directory; reuse it to resume")
    ap.add_argument("--benchmark-dir",
                    default=str(ROOT / "data" / "SOL-ExecBench" / "benchmark"),
                    type=Path)
    ap.add_argument("--harness", action="append", choices=sorted(HARNESSES),
                    help="repeatable; default is every harness")
    ap.add_argument("--categories", nargs="+",
                    help="default is all four; naming a subset is recorded")
    ap.add_argument("--limit-per-category", type=int,
                    help="sample evenly rather than taking the first N")
    ap.add_argument("--gpus", help="comma-separated torch indices; "
                                   "default is every GPU but the authoritative one")
    ap.add_argument("--max-attempts", type=int, default=5,
                    help="verification runs the agent may make per problem")
    ap.add_argument("--timeout-min", type=float, default=30.0,
                    help="wallclock cap per problem")
    ap.add_argument("--budget-usd", type=float,
                    help="stop starting new units once reported spend exceeds this")
    # No default, deliberately. `artifacts/05/workloads` is the MI350X tolerance
    # tree, and a default that silently resolves on the wrong part is the exact
    # failure this project keeps hitting: the run succeeds, every agent is judged
    # against another part's tolerances, and nothing in the output says so. On
    # MI355X the tree is artifacts/05-MI355X/workloads. Make the caller say which.
    ap.add_argument("--workloads-root", required=True, type=Path,
                    help="tree of AMD-derived tolerances FOR THIS PART, e.g. "
                         "artifacts/05-MI355X/workloads. Pass 'none' to opt into "
                         "the dataset's B200 tolerances as an explicit choice. "
                         "Required: there is no part-correct default.")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--retry-transient", action="store_true",
                    help="also re-run units whose recorded session failed on "
                         "infrastructure (gateway error, dropped stream) rather "
                         "than on the model's answer")
    ap.add_argument("--include-deferred", action="store_true",
                    help="also run problems listed in artifacts/deferred.json, "
                         "whose references do not run on ROCm at all")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the units and exit without spending anything")
    args = ap.parse_args()

    harnesses = args.harness or sorted(HARNESSES)
    workloads_root = None if str(args.workloads_root).lower() == "none" \
        else args.workloads_root

    deferred = {} if args.include_deferred else load_deferred(ROOT)
    problems, excluded = discover_problems(args.benchmark_dir, args.categories,
                                           args.limit_per_category, deferred)
    units = [Unit(harness=h, problem_dir=p) for h in harnesses for p in problems]

    run_root = ROOT / "artifacts" / "10" / "runs" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    else:
        import torch
        gpus = default_agent_gpus(torch.cuda.device_count())

    config = {
        "run_id": args.run_id,
        "harnesses": harnesses,
        "categories": args.categories or ["L1", "L2", "Quant", "FlashInfer-Bench"],
        "limit_per_category": args.limit_per_category,
        "problems": len(problems),
        "units": len(units),
        "gpus": gpus,
        "max_attempts": args.max_attempts,
        "timeout_min": args.timeout_min,
        "budget_usd": args.budget_usd,
        "workloads_root": str(workloads_root) if workloads_root else None,
        "tolerances": "amd-derived" if workloads_root else "dataset-shipped-b200",
        # Stated, not implied: a rate over 220 problems is not a rate over 235,
        # and the difference has to be visible wherever the rate is.
        "excluded_deferred": excluded,
        "excluded_reasons": {k: deferred[k] for k in excluded},
    }
    print(json.dumps(config, indent=2))

    if args.dry_run:
        for u in units:
            print(f"  {u.harness:<12} {u.problem_key}")
        return 0

    preflight(harnesses)

    # Written before the sweep, so an interrupted run still says what it was
    # asked to do -- which is the difference between a gap and a decision. The
    # digest is taken now so that scoring can tell whether the code that judges
    # correctness moved while agents were running.
    (run_root / "config.json").write_text(
        json.dumps(
            {**stamp("10-agents"), **config,
             "harness_tree_digest": tree_digest(ROOT)},
            indent=2, default=str,
        )
    )

    sweep = Sweep(
        run_root=run_root,
        harness_specs={h: {} for h in harnesses},
        gpus=gpus,
        max_attempts=args.max_attempts,
        timeout_s=int(args.timeout_min * 60),
        workloads_root=workloads_root,
        budget_usd=args.budget_usd,
        resume=not args.no_resume,
        retry_transient=args.retry_transient,
    )
    sweep.run(units)

    print(f"\nsessions under {run_root}")
    print(f"next: python scripts/score_solutions.py --run-id {args.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
