#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 02 runner — run a problem's own PyTorch reference as the solution.

Every problem should pass correctness against itself. A failure here is a ROCm
op-coverage gap or an input-generator breakage, never an optimization problem,
which is what makes this the acceptance criterion for the port: it separates
"the harness works on AMD" from "this kernel is slow on AMD".

    python scripts/runners/run_reference.py --problem data/.../L1/foo --out out.json

Dispatched per problem by `shard_sweep.py --task references`. On failure it
still writes its output file, with the error in it (see `_common.py`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
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
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    def body() -> dict:
        from sol_execbench.core import BenchmarkConfig

        definition, workloads = load_problem(a.problem)
        solution = reference_solution(definition)
        # Fewer iterations than the scoring default: this sweep answers "does
        # it run and is it correct", not "how fast". Timing quality here is
        # deliberately not load-bearing -- T_b is task 06, at full settings.
        config = BenchmarkConfig(
            warmup_runs=a.warmup,
            iterations=a.iterations,
            benchmark_reference=False,
        )
        traces = evaluate(definition, workloads, solution, config, timeout=a.timeout)
        result = summarize(traces)
        result.update({
            "problem": problem_key(a.problem),
            "definition": definition.name,
            "problem_dir": str(a.problem),
        })
        # `ok` means the runner completed, not that the problem passed. A
        # reference that fails correctness is a RESULT and must not read as a
        # crashed worker that shard_sweep would retry forever.
        result["ok"] = True
        return result

    return run_guarded(a.out, "02-reference-sweep", body)


if __name__ == "__main__":
    raise SystemExit(main())
