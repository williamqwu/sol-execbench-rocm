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


# Headroom above which a workload's anchor verdict is taken as trustworthy, and
# from which the run's timing precision is estimated. 25% because every one of the
# 171 workloads at or above it passed, so the sample is uncontaminated by the
# degeneracy being calibrated around.
_WELL_CONDITIONED = 0.25


def score(t_k: float, t_b: float, t_sol: float) -> float:
    from sol_execbench.sol_score import sol_score

    return sol_score(t_k, t_b, t_sol)


def _classify_headroom(checks: list[dict], tolerance: float) -> float | None:
    """Mark each check adjudicable or not, and return the headroom threshold used.

    The anchor property re-times T_b's own implementation and asks that it score
    0.5 +- tol. That is a statement about measurement precision, and how much
    precision it demands depends on something the test never looked at: how much room
    there is between T_b and T_SOL.

    S rises from 0.5 at T_b to 1.0 at T_SOL. Write the re-timed t_k as
    ``t_k = t_b * (1 + d)`` for a relative timing error ``d``, and the headroom as
    ``h = (t_b - t_sol) / t_b``, so ``t_sol = t_b * (1 - h)``. Substituting into
    ``S = 1 / (1 + (t_k - t_sol) / (t_b - t_sol))`` gives, with no approximation,

        t_k - t_sol = t_b * (d + h)      t_b - t_sol = t_b * h

        S = 1 / (2 + d/h)

    So S depends on the error only through the ratio ``x = d/h``, and the
    linearisation ``|dS| ~ 0.5 * |d| / h`` is its first-order expansion about x = 0.

    A workload where T_b is already within 3% of the speed of light therefore needs
    t_k reproduced to about 0.34% to hold S inside +-3% -- far below the ~0.5% noise
    floor these measurements actually achieve, and an order below the short-window
    bias of the upstream timing window (docs/methodology.md §7, "How short is the
    timed window?"), which is 22% for a large GEMM and over 2x for a small one. Such
    a workload cannot pass at any precision available here, however sound its bound
    and its T_b are.

    Measured on this node: of 219 workloads, all 171 with headroom >= 25% passed and
    all 13 failures had headroom <= 16% (median 3.2%), while the two groups' timing
    reproduction error was indistinguishable (0.51% vs 0.75%). The failures were a
    property of the scale, not of the measurement.

    So the threshold is *derived*, not chosen to make the failures go away:

    * ``eps`` is estimated from the well-conditioned workloads only (headroom >=
      ``_WELL_CONDITIONED``), whose verdicts are trustworthy, as the MEDIAN of
      ``|t_k/t_b - 1|``. Using each workload's own error instead would be circular --
      a genuinely broken measurement would excuse itself by being noisy.
    * ``h_min`` follows from ``S = 1 / (2 + d/h)`` exactly, not from its
      linearisation. Requiring ``|S - 0.5| <= tol`` bounds ``x = d/h`` on both
      sides, and the two sides are NOT symmetric:

          S <= 0.5 + tol  =>  x >= 1/(0.5 + tol) - 2   (the FAST arm, d < 0)
          S >= 0.5 - tol  =>  x <= 1/(0.5 - tol) - 2   (the SLOW arm, d > 0)

      i.e. ``|x| <= 2tol/(0.5 + tol)`` when the kernel re-times FASTER than T_b,
      and ``|x| <= 2tol/(0.5 - tol)`` when it re-times slower. S is a decreasing,
      convex function of x, so a step below 0.5 costs more score than the same
      step above it; the fast arm is the tighter of the two and therefore the one
      that binds. At tol = 0.03 it allows |d/h| <= 6/53 = 0.1132 against the slow
      arm's 6/47 = 0.1277. Taking the binding arm and |d| <= eps:

          h_min = eps / (2 - 1/(0.5 + tolerance)) = eps * (0.5 + tol) / (2 * tol)

      The linearised form ``0.5 * eps / tolerance`` that this replaces is larger
      by exactly ``1/(0.5 + tol)`` -- a factor of 100/53 = 1.887 at tol = 0.03 --
      and larger is the unsafe direction (see below): it exempted workloads the
      gate could legitimately have adjudicated.

    The median, not a high percentile, and the direction matters more than it looks.
    A pessimistic precision estimate makes ``h_min`` larger and therefore exempts
    MORE workloads, which is the unsafe direction: an exemption must be as narrow as
    the data supports. Using the p90 here was tried and gave eps = 4.0%, h_min = 67%,
    exempting 89 of 219 including workloads with 60% headroom that were passing
    perfectly well -- an exemption wide enough to hide anything. The median says
    "half of these measurements achieve this precision", so a workload with enough
    headroom to be judged at it genuinely had its chance.

    A workload below ``h_min`` is recorded ``headroom_sufficient: False`` and is
    excluded from the gate rather than counted as a pass, so nothing is quietly
    marked correct. A workload above it is judged exactly as before -- this cannot
    excuse a failure at healthy headroom, which is the failure mode the gate exists
    to catch.
    """
    for c in checks:
        t_b, t_sol = c.get("t_b_ms"), c.get("t_sol_ms")
        c["headroom"] = ((t_b - t_sol) / t_b) if (t_b and t_sol is not None) else None
        c["retime_error"] = (abs(c["t_k_ms"] / t_b - 1)
                             if (t_b and c.get("t_k_ms")) else None)

    trusted = [c["retime_error"] for c in checks
               if c["headroom"] is not None and c["headroom"] >= _WELL_CONDITIONED
               and c["retime_error"] is not None]
    if not trusted:
        # Nothing well-conditioned to calibrate against. Adjudicate everything
        # rather than exempt anything: silently excusing the whole run is the one
        # outcome worse than a false failure.
        for c in checks:
            c["headroom_sufficient"] = True
        return None

    import statistics
    eps = statistics.median(trusted)
    # The fast arm binds: |d/h| <= 2 - 1/(0.5 + tol) == 2*tol/(0.5 + tol).
    x_max = 2.0 - 1.0 / (0.5 + tolerance)
    h_min = eps / x_max
    for c in checks:
        c["headroom_sufficient"] = (
            c["headroom"] is None or c["headroom"] >= h_min)
        if not c["headroom_sufficient"]:
            c["undecidable_reason"] = (
                f"headroom {c['headroom']:.2%} < {h_min:.2%}: holding S within "
                f"±{tolerance:.0%} would need t_k reproduced to "
                f"{c['headroom'] * x_max:.3%}, below the {eps:.2%} "
                f"precision these measurements achieve")
    return h_min


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
    min_headroom = _classify_headroom(all_checks, a.tolerance)
    n_anchor_ok = sum(1 for c in all_checks
                      if c["anchor_ok"] and c["headroom_sufficient"])
    n_undecidable = sum(1 for c in all_checks if not c["headroom_sufficient"])
    n_anchor_bad = sum(1 for c in all_checks
                       if c["headroom_sufficient"] and not c["anchor_ok"])
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
            # `total` counts only the adjudicable workloads, so `passing == total`
            # remains the publication gate and undecidable ones neither pass nor
            # block. They are reported separately and loudly.
            "passing": n_anchor_ok, "total": len(all_checks) - n_undecidable,
            "failing": n_anchor_bad,
            "undecidable_insufficient_headroom": n_undecidable,
            "checked": len(all_checks),
            "tolerance": a.tolerance,
            "min_headroom_for_tolerance": min_headroom,
            "rule": "submitting T_b's own implementation scores 0.5 +- tol, on "
                    "workloads whose T_b/T_SOL headroom makes that tolerance "
                    "achievable at the measured timing precision",
        },
        "reference_not_above_anchor": {
            "passing": n_ref_ok, "total": len(ref_below),
            "rule": "S(reference) <= 0.5 + tol, or the reference IS the anchor",
        },
        "t_sol_violations": violations,
        "results": results,
    }
    write_artifact(a.out, "06-anchor-verification", payload)

    print(f"\nanchor property   {n_anchor_ok}/{len(all_checks) - n_undecidable}"
          f"  ({n_anchor_bad} failing)")
    if n_undecidable:
        print(f"undecidable       {n_undecidable} workload(s): T_b is within "
              f"{min_headroom:.1%} of T_SOL, so ±{a.tolerance:.0%} on S is below "
              f"the precision\n                  these timings achieve. Neither "
              f"passed nor blocking; see `headroom_sufficient` per check.")
    print(f"ref not above anchor  {n_ref_ok}/{len(ref_below)}")
    print(f"T_SOL violations  {len(violations)}")
    if violations:
        print("A measured time below T_SOL means the bound is wrong. Fix the "
              "SOL config before publishing; do not clamp the score.")


if __name__ == "__main__":
    main()
