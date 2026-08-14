#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarise the task-04 arm-order probe: does the divergence flip with order?

Reads the two directories `order_effect_probe.sh` writes and reports the median
divergence under each, paired per workload so the two orders are compared on
the same workloads and not on two different subsets.

Reading the result:

* **The medians are opposite in sign and similar in size** -> the bias is
  POSITIONAL. Whichever arm runs second reads slower, because under an unlocked
  clock basis it inherits a card the first arm heated. The divergence is then a
  property of the harness's duty cycle, not of either methodology.
* **Both medians are negative** -> `rocprof` reads low regardless of when it
  runs. That is a shim finding and a more serious one.

    python scripts/order_effect_report.py --probe /var/tmp/solbench/order-probe
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path


def load(d: Path) -> dict[tuple[str, str], tuple[float, bool]]:
    """{(problem, workload) -> (divergence_pct, microsecond_scale)}."""
    out: dict[tuple[str, str], tuple[float, bool]] = {}
    for f in sorted(glob.glob(f"{d}/*.json")):
        doc = json.loads(Path(f).read_text())
        if not doc.get("ok", True):
            continue
        for w in doc.get("divergences") or []:
            out[(doc["problem"], w["workload_uuid"])] = (
                w["divergence_pct"], w["microsecond_scale"])
    return out


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    help="also write the verdict as a provenance-stamped "
                         "artifact")
    a = ap.parse_args()

    A = load(a.probe / "A_hip_first")
    B = load(a.probe / "B_rocprof_first")
    shared = sorted(set(A) & set(B))
    if not shared:
        print("no paired workloads — did both orders run?")
        return 2

    print(f"paired workloads: {len(shared)}  "
          f"(A had {len(A)}, B had {len(B)})")
    print(f"problems: {len({p for p, _ in shared})}\n")

    groups = {
        "all workloads":     [k for k in shared],
        "kernels >= 100 us": [k for k in shared if not A[k][1]],
        "kernels < 100 us":  [k for k in shared if A[k][1]],
    }
    print(f"{'group':<20} {'n':>5} {'A hip_first':>13} "
          f"{'B rocprof_first':>16} {'flip?':>7}")
    for name, keys in groups.items():
        if not keys:
            continue
        ma, mb = med([A[k][0] for k in keys]), med([B[k][0] for k in keys])
        flip = "YES" if ma * mb < 0 else "no"
        print(f"{name:<20} {len(keys):>5} {ma:>+12.2f}% {mb:>+15.2f}% "
              f"{flip:>7}")

    # The positional reading, stated as a number: if the bias belongs to the
    # slot rather than to the methodology, then "whatever ran second" is slower
    # by about the same amount in both orders.
    print()
    second_pen = []
    for k in shared:
        # A: hip first, rocprof second. divergence = (hip - roc)/hip, so a
        # slower SECOND arm makes this negative.
        # B: rocprof first, hip second. A slower second arm makes it positive.
        second_pen.append(-A[k][0])
        second_pen.append(+B[k][0])
    print(f"median penalty on whichever arm ran SECOND: "
          f"{med(second_pen):+.2f}%  (n={len(second_pen)})")

    # And the order-free estimate: average the two orders per workload, which
    # cancels a symmetric positional term and leaves the methodology difference.
    resid = [(A[k][0] + B[k][0]) / 2 for k in shared]
    resid_big = [(A[k][0] + B[k][0]) / 2 for k in shared if not A[k][1]]
    print(f"order-cancelled divergence (mean of the two orders), "
          f"all: {med(resid):+.2f}%   >=100us: {med(resid_big):+.2f}%")

    if a.out:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from provenance import write_artifact

        ma = {n: med([A[k][0] for k in ks]) for n, ks in groups.items() if ks}
        mb = {n: med([B[k][0] for k in ks]) for n, ks in groups.items() if ks}
        flipped = any(ma[n] * mb[n] < 0 for n in ma)
        write_artifact(a.out, "04-arm-order-probe", {
            "paired_workloads": len(shared),
            "problems": sorted({p for p, _ in shared}),
            "median_A_hip_first": ma,
            "median_B_rocprof_first": mb,
            "median_second_slot_penalty_pct": med(second_pen),
            "order_cancelled_median_pct": {
                "all workloads": med(resid), "kernels >= 100 us": med(resid_big)},
            "any_group_flipped_sign": flipped,
            "verdict": ("ORDER EFFECT CONFIRMED — the bias follows the slot, "
                        "not the methodology"
                        if flipped else
                        "ORDER EFFECT NOT CONFIRMED — the divergence keeps its "
                        "sign in both orders, so it is a property of the "
                        "methodology pair and not of the card's thermal state "
                        "when each arm ran"),
        })
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
