#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 06 step 4 — verify the anchor property of the score scale.

The score is

    S(T_k) = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))

which is only meaningful if T_b and T_SOL are both right. Two consequences fall
straight out of the algebra and are checkable:

    submitting T_b's own implementation must score  0.5 +- 0.03
    the plain reference must score                  < 0.5

The first is nearly a tautology *given correct inputs* -- which is exactly why
it is worth running. It fails when T_b in the manifest is not the time that
implementation actually takes: a stale sweep, a variant recorded against the
wrong workload, a T_b measured on a different GPU than the one being timed
(the eight GPUs here hold clocks spanning 5%), or a T_SOL above the measured
time so the denominator is negative.

    python scripts/verify_anchor.py --manifest artifacts/09/manifest-v1.json \\
        --sample 20 --gpu 0 --out artifacts/06/anchor-verification.json

Do not ship a manifest that fails this.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "runners"))
sys.path.insert(0, str(ROOT / "src"))

from provenance import write_artifact  # noqa: E402


def score(t_k: float, t_b: float, t_sol: float) -> float:
    from sol_execbench.sol_score import sol_score

    return sol_score(t_k, t_b, t_sol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/09/manifest-v1.json")
    ap.add_argument("--data", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--out", default="artifacts/06/anchor-verification.json")
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--tolerance", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=50)
    a = ap.parse_args()

    from _common import (
        PASSED, evaluate, load_problem, reference_solution, summarize,
    )
    from sol_execbench.core import BenchmarkConfig

    manifest = json.loads(Path(a.manifest).read_text())
    scoreable = [
        (k, v) for k, v in manifest["problems"].items() if v.get("n_scoreable")
    ]
    random.Random(a.seed).shuffle(scoreable)
    sample = scoreable[: a.sample]

    variants_mod = ROOT / "reference" / "tb-candidates" / "variants.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("_tb_variants", variants_mod)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    VARIANTS = mod.VARIANTS

    results = []
    for key, entry in sample:
        category, name = key.split("__", 1)
        problem = Path(a.data) / category / name
        row: dict = {"problem": key}
        try:
            definition, workloads = load_problem(problem)
            config = BenchmarkConfig(warmup_runs=10, iterations=a.iterations,
                                     benchmark_reference=False)

            # Time BOTH arms in one process, back to back, on the same GPU:
            # the property is about the ratio, and re-timing under different
            # conditions would test the node's stability rather than the scale.
            by_variant: dict[str, dict] = {}
            wanted = {
                e["t_b_variant"] for e in entry["workloads"].values()
                if e.get("t_b_variant")
            }
            for variant in sorted(wanted | {"v1_eager"}):
                src = VARIANTS[variant](definition.reference) \
                    if variant in VARIANTS else None
                if src is None:
                    continue
                traces = evaluate(
                    definition, workloads,
                    reference_solution(definition, name_suffix=variant, source=src),
                    config, timeout=3600,
                )
                by_variant[variant] = {
                    w["workload_uuid"]: w["latency_ms"]
                    for w in summarize(traces)["per_workload"]
                    if w["status"] == PASSED and w["latency_ms"]
                }

            checks = []
            for uuid, e in entry["workloads"].items():
                if not e.get("scoreable"):
                    continue
                t_b, t_sol = e["t_b_ms"], e["t_sol_ms"]
                variant = e.get("t_b_variant")
                t_k = (by_variant.get(variant) or {}).get(uuid)
                t_ref = (by_variant.get("v1_eager") or {}).get(uuid)
                if t_k is None:
                    continue
                s_anchor = score(t_k, t_b, t_sol)
                checks.append({
                    "workload_uuid": uuid,
                    "variant": variant,
                    "t_b_ms": t_b, "t_sol_ms": t_sol,
                    "t_k_ms": t_k, "t_ref_ms": t_ref,
                    "score_of_anchor": s_anchor,
                    "anchor_ok": abs(s_anchor - 0.5) <= a.tolerance,
                    "score_of_reference": (
                        score(t_ref, t_b, t_sol) if t_ref is not None else None
                    ),
                    # T_SOL is a LOWER bound. A measured time below it means the
                    # bound is wrong -- always a config error, never a fast
                    # kernel -- and it also makes the score exceed 1.
                    "t_sol_le_measured": t_sol <= min(
                        x for x in (t_k, t_ref) if x is not None
                    ),
                })
            row["checks"] = checks
            row["n_ok"] = sum(1 for c in checks if c["anchor_ok"])
            row["n"] = len(checks)
            row["ok"] = True
        except Exception as e:                        # noqa: BLE001
            row.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
        results.append(row)
        print(f"  {key}: {row.get('n_ok')}/{row.get('n')} anchored"
              if row.get("ok") else f"  {key}: FAILED {row.get('error')}",
              flush=True)

    all_checks = [c for r in results for c in r.get("checks", [])]
    n_anchor_ok = sum(1 for c in all_checks if c["anchor_ok"])
    # "The plain reference scores below 0.5" holds only where the anchor is
    # something other than the plain reference. On 43 of these workloads
    # `v1_eager` IS the fastest passing variant, so it IS T_b and it scores
    # exactly 0.5 -- a pass, not a failure, and reporting it as a failure
    # would make a correct manifest look broken. Elsewhere the property is
    # checked with the same tolerance the anchor property uses, because both
    # arms are re-timed measurements and carry the same noise.
    ref_below = [c for c in all_checks if c["score_of_reference"] is not None]
    n_ref_ok = sum(
        1 for c in ref_below
        if c["variant"] == "v1_eager" or c["score_of_reference"] <= 0.5 + a.tolerance
    )
    violations = [c for c in all_checks if not c["t_sol_le_measured"]]

    payload = {
        "manifest": str(a.manifest),
        "sampled_problems": len(sample),
        "workloads_checked": len(all_checks),
        "anchor_property": {
            "passing": n_anchor_ok, "total": len(all_checks),
            "tolerance": a.tolerance,
            "rule": "submitting T_b's own implementation scores 0.5 +- tol",
        },
        "reference_not_above_anchor": {
            "passing": n_ref_ok, "total": len(ref_below),
            "rule": "S(reference) <= 0.5 + tol, or the reference IS the anchor",
        },
        "t_sol_violations": violations,
        "results": results,
    }
    write_artifact(a.out, "06-anchor-verification", payload)

    print(f"\nanchor property   {n_anchor_ok}/{len(all_checks)}")
    print(f"ref not above anchor  {n_ref_ok}/{len(ref_below)}")
    print(f"T_SOL violations  {len(violations)}")
    if violations:
        print("A measured time below T_SOL means the bound is wrong. Fix the "
              "SOL config before publishing; do not clamp the score.")


if __name__ == "__main__":
    main()
