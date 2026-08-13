#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read `clock_ab.py`'s rows and say what the clock policy actually cost.

Two questions, and they are not the same question:

* **Speed.** Per problem, the median across blocks of each condition, and the
  ratio against the node's standing policy (`locked1600`). A ratio above 1
  means the locked node is SLOWER, which is the cost of locking.
* **Stability.** Locking is bought for stability, so a comparison that reports
  only speed cannot price it. Two spreads are reported and they measure
  different things: the within-process CV over reps, and the between-block
  spread of the per-block medians, which is the one that survives a fresh
  process, a re-warm and several minutes of drift.

The **noise floor is measured, not assumed**: `locked1600` appears in every
block, so the spread of that condition against itself is what any difference
between conditions has to beat.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--baseline", default="locked1600")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    rows = json.loads(a.rows.read_text())
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("ok") and r.get("ms_per_call_p50"):
            by[(r["problem"], r["condition"])].append(r)

    problems = sorted({p for p, _ in by})
    conditions = sorted({c for _, c in by})
    if a.baseline in conditions:
        conditions = [a.baseline] + [c for c in conditions if c != a.baseline]

    out = {"baseline": a.baseline, "conditions": conditions, "problems": {}}
    print(f"{'problem':<52} {'condition':<12} {'ms p50':>10} {'vs base':>8} "
          f"{'MHz':>6} {'W':>6} {'blk spread':>10} {'rep CV':>8}")
    for p in problems:
        base = None
        for c in conditions:
            rs = by.get((p, c)) or []
            if not rs:
                continue
            ms = [r["ms_per_call_p50"] for r in rs]
            med = statistics.median(ms)
            if c == a.baseline:
                base = med
            spread = (max(ms) - min(ms)) / med if len(ms) > 1 and med else 0.0
            mhz = statistics.median([r["gfx_mhz_p50"] for r in rs if r.get("gfx_mhz_p50")] or [0])
            watt = statistics.median([r["power_w_mean"] for r in rs if r.get("power_w_mean")] or [0])
            cv = statistics.median([r["ms_per_call_cv"] for r in rs if r.get("ms_per_call_cv") is not None] or [0])
            ratio = (med / base) if base else float("nan")
            out["problems"].setdefault(p, {})[c] = {
                "n_blocks": len(rs), "ms_p50": med, "ms_all_blocks": ms,
                "ratio_vs_baseline": ratio, "block_spread_frac": spread,
                "gfx_mhz_p50": mhz, "power_w": watt, "rep_cv_p50": cv,
            }
            print(f"{p[:52]:<52} {c:<12} {med:>10.4f} {ratio:>8.3f} "
                  f"{mhz:>6.0f} {watt:>6.0f} {spread*100:>9.2f}% {cv*100:>7.3f}%")
        print()

    # Aggregate: the geometric mean of the per-problem ratios is the honest
    # headline, because a ratio averaged arithmetically is biased by whichever
    # direction happens to be the numerator.
    print(f"{'AGGREGATE':<52} {'condition':<12} {'geomean':>10} {'min':>8} {'max':>8} "
          f"{'median blk spread':>18}")
    agg = {}
    for c in conditions:
        ratios, spreads = [], []
        for p in problems:
            e = out["problems"].get(p, {}).get(c)
            if e and e["ratio_vs_baseline"] == e["ratio_vs_baseline"]:
                ratios.append(e["ratio_vs_baseline"])
                spreads.append(e["block_spread_frac"])
        if not ratios:
            continue
        gm = statistics.geometric_mean(ratios)
        agg[c] = {"geomean_ratio": gm, "min_ratio": min(ratios),
                  "max_ratio": max(ratios), "n_problems": len(ratios),
                  "median_block_spread_frac": statistics.median(spreads)}
        print(f"{'':<52} {c:<12} {gm:>10.4f} {min(ratios):>8.3f} {max(ratios):>8.3f} "
              f"{statistics.median(spreads)*100:>17.2f}%")
    out["aggregate"] = agg

    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
