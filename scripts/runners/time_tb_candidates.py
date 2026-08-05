#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 06 runner — time every T_b candidate variant for one problem.

T_b is the optimized-PyTorch anchor that sets the entire score scale
(S = 0.5 at T_k = T_b), so it is measured, per platform, under the harness's
own conditions — not ported from B200 and not hand-tuned to make scores land
somewhere pleasing.

Variants come from `reference/tb-candidates/variants.py` (generic transforms of
the problem's own reference) plus any per-problem overrides in
`reference/tb-candidates/<Category>__<problem>/vN_*.py`. The winner is the
fastest variant that is also CORRECT: a variant that is fast because it is
wrong is not a baseline.

    python scripts/runners/time_tb_candidates.py --problem <dir> --out <file>
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    PASSED,
    ROOT,
    evaluate,
    load_problem,
    problem_key,
    reference_solution,
    run_guarded,
    summarize,
    workloads_path,
)
from provenance import ClockMonitor, clock_basis  # noqa: E402

CANDIDATE_DIR = ROOT / "reference" / "tb-candidates"


@contextlib.contextmanager
def _variant_clock():
    """Sample the clock across ONE variant, on the unlocked basis only.

    `run_guarded` already samples across the whole runner, which is the right
    granularity for a fixed F_LOCK: one setpoint covers every variant, so a single
    verdict on "was the node at F_LOCK" applies to all of them.

    It is the wrong granularity unlocked, because there the clock is a property of
    the kernel. Variants of one problem are different kernels with different power
    draws -- an eager formulation and a `max_autotune` one need not run at the same
    frequency -- so a median across all of them describes none, and would trip the
    12% spread gate on nearly every problem. T_SOL has to be evaluated at the clock
    of the variant that actually won.

    Yields None under the fixed basis so the surrounding code stays one path, and so
    the sampler's own amdsmi polling is not added to runs that have no use for it.
    """
    if clock_basis() != "unlocked":
        yield None
        return
    monitor = ClockMonitor()
    with monitor:
        yield monitor


def _load_variants() -> dict:
    spec = importlib.util.spec_from_file_location(
        "_tb_variants", CANDIDATE_DIR / "variants.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.VARIANTS)


def _overrides(key: str, reference_src: str) -> dict[str, str]:
    """Hand-written per-problem variants, if any.

    The generic set is the default because task 06 is explicitly a batch job,
    not an authoring one. An override exists only where the pre-authored set
    was clearly missing an obvious formulation, and its presence in the
    artifact is how that addition gets recorded.
    """
    d = CANDIDATE_DIR / key
    if not d.is_dir():
        return {}
    return {p.stem: p.read_text() for p in sorted(d.glob("v*_*.py"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--only-variant", action="append", default=None,
                    help="restrict to named variants (the authoritative pass "
                         "re-times only the winner)")
    a = ap.parse_args()

    def body() -> dict:
        from sol_execbench.core import BenchmarkConfig

        definition, workloads = load_problem(a.problem)
        key = problem_key(a.problem)

        sources = {
            name: transform(definition.reference)
            for name, transform in _load_variants().items()
        }
        overrides = _overrides(key, definition.reference)
        sources.update(overrides)
        if a.only_variant:
            sources = {k: v for k, v in sources.items() if k in set(a.only_variant)}

        config = BenchmarkConfig(
            warmup_runs=a.warmup,
            iterations=a.iterations,
            benchmark_reference=False,
        )

        results: dict[str, dict] = {}
        for name, src in sources.items():
            variant_clock = None
            try:
                with _variant_clock() as monitor:
                    traces = evaluate(
                        definition,
                        workloads,
                        reference_solution(definition, name_suffix=name, source=src),
                        config,
                        timeout=a.timeout,
                    )
                    summary = summarize(traces)
                if monitor is not None:
                    variant_clock = monitor.summary()
            except Exception as e:                  # noqa: BLE001
                # A variant that fails to compile or run is a RESULT: some
                # formulations legitimately do not work for some problems
                # (torch.compile falls over on a few). Recording it keeps the
                # remaining variants' timings, which is the point of doing
                # this per variant rather than per problem.
                results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                continue

            per_wl = {
                w["workload_uuid"]: w["latency_ms"]
                for w in summary["per_workload"]
                if w["status"] == PASSED and w["latency_ms"]
            }
            results[name] = {
                "ok": True,
                "workloads": summary["workloads"],
                "passed": summary["passed"],
                "all_passed": summary["all_passed"],
                "latency_ms_by_workload": per_wl,
                "is_override": name in overrides,
                # None under the fixed basis, where run_guarded's node-wide verdict
                # is what applies.
                "measured_clock": variant_clock,
                "failures": [
                    {"workload_uuid": w["workload_uuid"], "status": w["status"],
                     "log": w["log"][:1000]}
                    for w in summary["per_workload"] if w["status"] != PASSED
                ],
            }

        # -- Select the winner PER WORKLOAD ------------------------------------
        # Per workload, not per problem: the fastest formulation genuinely
        # differs with shape (compile wins on large shapes, eager on tiny ones
        # where compilation guards dominate), and T_b is defined per workload
        # instance. Picking one variant for the whole problem would inflate
        # T_b on every shape it does not suit, which would make every score on
        # that shape look better than it is.
        winners: dict[str, dict] = {}
        for name, r in results.items():
            if not r.get("ok") or not r.get("all_passed"):
                continue
            for uuid, ms in r["latency_ms_by_workload"].items():
                if uuid not in winners or ms < winners[uuid]["t_b_ms"]:
                    winners[uuid] = {"variant": name, "t_b_ms": ms}
                    # The winning variant's clock travels WITH its T_b, because that
                    # pairing is what the manifest anchors a score on: unlocked, a
                    # T_b without the frequency it was taken at cannot be turned
                    # into a bound. Absent under the fixed basis, where F_LOCK
                    # applies to every artifact alike.
                    mc = r.get("measured_clock")
                    if mc is not None:
                        winners[uuid]["f_for_bound_mhz"] = mc.get("f_for_bound_mhz")
                        winners[uuid]["clock_stable"] = mc.get("clock_stable")

        return {
            "problem": key,
            "definition": definition.name,
            # Which tolerances decided `all_passed`, and therefore which
            # variant was eligible to become the anchor.
            "workloads_from": str(workloads_path(a.problem)),
            "gpu": os.environ.get("HIP_VISIBLE_DEVICES"),
            "variants": results,
            "winner_by_workload": winners,
            "n_workloads": len(workloads),
            "n_workloads_with_tb": len(winners),
            # Loud rather than silent: a problem where no variant passed every
            # workload has no anchor, and must reach triage instead of being
            # quietly absent from the manifest.
            "complete": len(winners) == len(workloads),
        }

    return run_guarded(a.out, "06-tb-candidates", body)


if __name__ == "__main__":
    raise SystemExit(main())
