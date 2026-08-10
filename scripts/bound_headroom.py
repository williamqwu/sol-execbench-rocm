#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The defect class nothing checks: a T_SOL far BELOW anything achievable.

    python scripts/bound_headroom.py --manifest artifacts/09/manifest-v1.2.json \
        --out artifacts/11/bound-headroom.json

The board checks one invariant on a bound: no measurement may beat it. That
catches a T_SOL that is too LARGE, and it is the only automatic check there is.
A T_SOL that is too SMALL breaks no invariant at all -- it is a valid lower
bound, just a uselessly weak one -- so nothing anywhere reports it.

It is not harmless. The score is

    S = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))

and as `T_SOL -> 0` that becomes `T_b / (T_b + T_k)`: a comparison against the
PyTorch anchor with no roofline content left in it. Every problem in that
regime is scored on a different question from the rest of the board, and its
scores cluster near 0.5 whatever the kernel does.

`L2__036` is the case that surfaced this. Its `T_b/T_SOL` is 166 to 6,280
across its workloads -- its reference runs a 128-iteration Python loop of
unfold-and-reduce that SOLAR's graph barely sees -- and it has never appeared
on any defect list, because a bound nothing can reach is a bound nothing can
contradict.

Headroom `T_b / T_SOL` is the measure, not `T_k / T_SOL`: `T_b` is a property
of the problem and the part, so the figure does not change when a better kernel
arrives. There is no correct threshold and this script does not invent one; it
reports the distribution and lets the tail be read. For orientation, the score
a perfect kernel could earn is bounded by how much of the range is real, and at
headroom 100x a kernel would have to beat `T_b` by 50x to reach S = 0.99.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import write_artifact  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "artifacts" / "09" / "manifest-v1.2.json")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "11" / "bound-headroom.json")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    man = json.loads(a.manifest.read_text())
    per_problem = []
    all_h: list[float] = []

    for key, prob in man["problems"].items():
        hs, srcs, botts = [], set(), set()
        for w in prob.get("workloads", {}).values():
            t_sol, t_b = w.get("t_sol_ms"), w.get("t_b_ms")
            if not t_sol or not t_b or t_sol <= 0:
                continue
            hs.append(t_b / t_sol)
            srcs.add(w.get("t_sol_source"))
            botts.add(w.get("sol_bottleneck"))
        if not hs:
            continue
        all_h.extend(hs)
        per_problem.append({
            "problem": key,
            "category": prob.get("category"),
            "n_workloads": len(hs),
            "headroom_median": statistics.median(hs),
            "headroom_min": min(hs),
            "headroom_max": max(hs),
            "t_sol_sources": sorted(x for x in srcs if x),
            "bottlenecks": sorted(x for x in botts if x),
        })

    per_problem.sort(key=lambda r: r["headroom_median"], reverse=True)
    all_h.sort()

    def q(p: float) -> float:
        return all_h[min(len(all_h) - 1, int(p * len(all_h)))]

    bands = {
        "under_2x": sum(1 for h in all_h if h < 2),
        "2x_to_10x": sum(1 for h in all_h if 2 <= h < 10),
        "10x_to_100x": sum(1 for h in all_h if 10 <= h < 100),
        "100x_to_1000x": sum(1 for h in all_h if 100 <= h < 1000),
        "over_1000x": sum(1 for h in all_h if h >= 1000),
    }

    doc = {
        "question": (
            "How many bounds are so far below the measured anchor that the "
            "score they produce carries no roofline information?"
        ),
        "measure": "headroom = T_b / T_SOL, per workload",
        "manifest": str(a.manifest.relative_to(ROOT)),
        "manifest_version": man.get("manifest_version"),
        "workloads": len(all_h),
        "quantiles": {"p10": q(0.10), "p50": q(0.50), "p90": q(0.90),
                      "p99": q(0.99), "max": all_h[-1]},
        "bands": bands,
        "worst_problems": per_problem[:a.top],
        "note": (
            "No threshold is asserted here. A large headroom is not by itself "
            "an error -- a genuinely memory-bound problem whose reference is a "
            "poor implementation will show one honestly. What it does mean is "
            "that the score is dominated by T_b, and that no automatic check "
            "in this repo looks at it."
        ),
    }

    print(f"manifest {doc['manifest_version']}  workloads {len(all_h)}")
    print(f"headroom T_b/T_SOL   p10 {q(0.10):.2f}  p50 {q(0.50):.2f}  "
          f"p90 {q(0.90):.1f}  p99 {q(0.99):.1f}  max {all_h[-1]:.1f}")
    for k, v in bands.items():
        print(f"  {k:16} {v:5d}  ({100*v/len(all_h):.1f}%)")
    print(f"\nworst {a.top} problems by median headroom:")
    for r in per_problem[:a.top]:
        print(f"  {r['headroom_median']:10.1f}x  {r['problem'][:62]:62} "
              f"{','.join(r['bottlenecks'])}")

    write_artifact(a.out, "11-bound-headroom", doc)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
