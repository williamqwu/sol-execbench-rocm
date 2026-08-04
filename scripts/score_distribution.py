#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 09 — what scores does the manifest actually produce?

A manifest can be internally consistent and still put every real kernel at
S = 0.02 or S = 0.99. The only way to know is to score something real against
it, so this scores every variant the authoritative pass timed on GPU 0 — the
same measurements T_b was chosen from, now read as submissions.

**This is not upstream's agent baseline.** Upstream ran a kernel-optimizing
agent over the problem set and reported a median of 0.732. Nothing here ran an
agent; what is scored is a fixed set of PyTorch formulations (eager, compiled,
max-autotune, contiguous). The distribution is therefore a property of the
*scale*, not a claim about how well agents do on AMD, and it is labelled that
way everywhere it appears.

What it does establish, and what the agent baseline would also have to satisfy:

  * every score is finite and in (0, 1]
  * S = 0.5 lands on the variant that became T_b, by construction -- a
    tautology that fails loudly whenever T_b in the manifest is not the time
    that implementation actually takes
  * the relationship between S and the fraction of headroom reclaimed,
    `(T_ref - T_k) / (T_ref - T_SOL)`, which upstream reports at r = 0.981.
    Here T_ref is the problem's own reference, which is exactly what
    `v1_eager` runs.

    python scripts/score_distribution.py --out artifacts/09/score-distribution.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import write_artifact  # noqa: E402

REFERENCE_VARIANT = "v1_eager"


def sol_score(t_k: float, t_b: float, t_sol: float) -> float | None:
    denom = t_b - t_sol
    if denom <= 0:
        return None
    return 1.0 / (1.0 + (t_k - t_sol) / denom)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/09/manifest-v1.json")
    ap.add_argument("--authoritative", default="artifacts/06/authoritative")
    ap.add_argument("--out", default="artifacts/09/score-distribution.json")
    a = ap.parse_args()

    manifest = json.loads(Path(a.manifest).read_text())
    mprob = manifest["problems"]

    scores: list[float] = []
    anchor_scores: list[float] = []
    head_x: list[float] = []
    head_y: list[float] = []
    by_variant: dict[str, list[float]] = {}
    per_workload: dict[tuple, list[tuple[float, float]]] = {}
    n_pairs = n_unscoreable = 0

    for f in sorted(glob.glob(f"{a.authoritative}/*.json")):
        if f.endswith("no-winner.json"):
            continue
        doc = json.loads(Path(f).read_text())
        key = doc.get("problem") or Path(f).stem
        entry = (mprob.get(key) or {}).get("workloads") or {}
        variants = doc.get("variants") or {}
        winners = doc.get("winner_by_workload") or {}
        ref_lat = ((variants.get(REFERENCE_VARIANT) or {})
                   .get("latency_ms_by_workload") or {})

        for name, r in variants.items():
            if not r.get("ok"):
                continue
            for u, t_k in (r.get("latency_ms_by_workload") or {}).items():
                w = entry.get(u) or {}
                t_sol, t_b = w.get("t_sol_ms"), w.get("t_b_ms")
                if t_sol is None or t_b is None:
                    n_unscoreable += 1
                    continue
                s = sol_score(t_k, t_b, t_sol)
                if s is None:
                    n_unscoreable += 1
                    continue
                n_pairs += 1
                scores.append(s)
                by_variant.setdefault(name, []).append(s)
                if (winners.get(u) or {}).get("variant") == name:
                    anchor_scores.append(s)
                t_ref = ref_lat.get(u)
                if t_ref and t_ref > t_sol:
                    h = (t_ref - t_k) / (t_ref - t_sol)
                    head_x.append(h)
                    head_y.append(s)
                    per_workload.setdefault((key, u), []).append((h, s))

    within = [r for pts in per_workload.values()
              if len(pts) >= 3
              for r in [pearson([x for x, _ in pts], [y for _, y in pts])]
              if r is not None]
    within_median = statistics.median(within) if within else None

    def summary(xs: list[float]) -> dict:
        if not xs:
            return {}
        s = sorted(xs)
        return {
            "n": len(s), "min": s[0], "max": s[-1],
            "p10": s[len(s) // 10], "median": statistics.median(s),
            "p90": s[9 * len(s) // 10], "mean": statistics.fmean(s),
        }

    anchor = summary(anchor_scores)
    payload = {
        "_note": "Scores of the T_b variant set against the frozen manifest. "
                 "NOT an agent baseline -- no agent was run. See the script "
                 "docstring.",
        "manifest_version": manifest.get("manifest_version"),
        "all_variants": summary(scores),
        "by_variant": {k: summary(v) for k, v in sorted(by_variant.items())},
        "anchor_check": {
            **anchor,
            "_note": "The variant that became T_b, scored against it. Must be "
                     "0.5 by construction; a deviation means the manifest's "
                     "T_b is not the time that implementation takes.",
            "max_abs_deviation_from_half": (
                max(abs(x - 0.5) for x in anchor_scores) if anchor_scores else None),
        },
        "headroom_correlation": {
            "n": len(head_x),
            "pearson_r_pooled": pearson(head_x, head_y),
            "within_workload_r_median": within_median,
            "within_workload_n": len(within),
            "headroom_span_covered": (
                [min(head_x), max(head_x)] if head_x else None),
            "definition": "x = (T_ref - T_k)/(T_ref - T_SOL) with T_ref the "
                          "problem's own reference (v1_eager); y = S(T_k).",
            "_not_comparable_to_upstream":
                "Upstream reports r = 0.981 over AGENT submissions, which span "
                "the performance range. These submissions are four PyTorch "
                "formulations of the same reference, and T_b is the fastest of "
                "them, so they cluster: the pooled correlation is computed "
                "over a sample that barely varies and is not a replication of "
                "that result. The within-workload figure is the one this data "
                "can support. A real replication needs the agent baseline, "
                "which was not run -- see the release notes.",
        },
        "workload_variant_pairs": n_pairs,
        "pairs_not_scoreable": n_unscoreable,
    }
    write_artifact(Path(a.out), "09-score-distribution", payload)

    print(f"score distribution -> {a.out}")
    d = payload["all_variants"]
    if d:
        print(f"  all variants   n={d['n']}  median {d['median']:.3f}  "
              f"p10 {d['p10']:.3f}  p90 {d['p90']:.3f}")
    if anchor:
        print(f"  anchor check   n={anchor['n']}  median {anchor['median']:.4f}  "
              f"max |S-0.5| = "
              f"{payload['anchor_check']['max_abs_deviation_from_half']:.2e}")
    hc = payload["headroom_correlation"]
    r = hc["pearson_r_pooled"]
    print(f"  headroom r     pooled {r:.4f} over {len(head_x)} pairs"
          if r is not None else "  headroom r     n/a")
    if within_median is not None:
        print(f"                 within-workload median {within_median:.4f} "
              f"over {len(within)} workloads")
    print(f"  not scoreable  {n_unscoreable} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
