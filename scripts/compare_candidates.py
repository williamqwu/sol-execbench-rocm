#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What the recompile-limit fix (STATE.md D50) changed, per problem and variant.

Compares two task-06 candidate directories -- the shipped one and a re-run --
on the three things the fix was predicted to move:

* **Pass counts go DOWN for the compile variants.** Before the fix, a workload
  past dynamo's 8th distinct shape ran eagerly and was recorded as a compile
  pass. Those are the false passes; a correct re-run must lose them.
* **Latencies for the surviving compile passes change**, because the ones that
  were eager are now actually compiled.
* **The per-workload winner changes**, which is what moves T_b, which is what
  moves every score.

Nothing here re-times anything. It reads two sets of artifacts.

    python scripts/compare_candidates.py \\
        --before artifacts/06/candidates \\
        --after  artifacts/12/tb-recompile-fix/candidates \\
        --out    artifacts/12/tb-recompile-fix/comparison.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

COMPILE_VARIANTS = {"v2_compile", "v3_compile_max_autotune", "v5_compile_contiguous"}


def load(d: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(d.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text())
        except json.JSONDecodeError:
            out[f.stem] = {"ok": False, "error": "unreadable artifact"}
    return out


def variant_rows(payload: dict) -> dict[str, dict]:
    return payload.get("variants") or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, type=Path)
    ap.add_argument("--after", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    before, after = load(a.before), load(a.after)
    keys = sorted(set(before) & set(after))
    only_before = sorted(set(before) - set(after))
    only_after = sorted(set(after) - set(before))

    per_variant = defaultdict(lambda: {
        "problems_compared": 0, "passed_before": 0, "passed_after": 0,
        "problems_losing_passes": [], "problems_gaining_passes": [],
        "errored_before": 0, "errored_after": 0,
    })
    latency_ratios: dict[str, list[float]] = defaultdict(list)
    winner_changes = []
    problems = []

    for key in keys:
        vb, va = variant_rows(before[key]), variant_rows(after[key])
        row = {"problem": key, "variants": {}}
        for name in sorted(set(vb) | set(va)):
            b, c = vb.get(name, {}), va.get(name, {})
            agg = per_variant[name]
            agg["problems_compared"] += 1
            agg["errored_before"] += int(b.get("ok") is False)
            agg["errored_after"] += int(c.get("ok") is False)
            pb, pa = b.get("passed"), c.get("passed")
            if pb is None or pa is None:
                continue
            agg["passed_before"] += pb
            agg["passed_after"] += pa
            if pa < pb:
                agg["problems_losing_passes"].append([key, pb, pa])
            elif pa > pb:
                agg["problems_gaining_passes"].append([key, pb, pa])

            lb = b.get("latency_ms_by_workload") or {}
            la = c.get("latency_ms_by_workload") or {}
            shared = [u for u in lb if u in la and lb[u] and la[u]]
            ratios = [la[u] / lb[u] for u in shared]
            if ratios:
                latency_ratios[name].extend(ratios)
            row["variants"][name] = {
                "passed_before": pb, "passed_after": pa,
                "workloads": c.get("workloads"),
                "n_shared_timed": len(shared),
                "latency_ratio_p50": statistics.median(ratios) if ratios else None,
                "latency_ratio_min": min(ratios) if ratios else None,
                "latency_ratio_max": max(ratios) if ratios else None,
            }

        wb = before[key].get("winner_by_workload") or {}
        wa = after[key].get("winner_by_workload") or {}
        changed = []
        for u in sorted(set(wb) & set(wa)):
            nb = wb[u].get("variant") if isinstance(wb[u], dict) else wb[u]
            na = wa[u].get("variant") if isinstance(wa[u], dict) else wa[u]
            if nb != na:
                changed.append([u, nb, na])
        if changed:
            winner_changes.append({"problem": key, "n_changed": len(changed),
                                   "n_workloads": len(set(wb) & set(wa)),
                                   "examples": changed[:5]})
        row["winners_changed"] = len(changed)
        row["winners_compared"] = len(set(wb) & set(wa))
        problems.append(row)

    summary = {}
    for name, agg in sorted(per_variant.items()):
        r = latency_ratios.get(name) or []
        summary[name] = {
            **{k: v for k, v in agg.items()
               if k not in ("problems_losing_passes", "problems_gaining_passes")},
            "passes_lost": agg["passed_before"] - agg["passed_after"],
            "n_problems_losing_passes": len(agg["problems_losing_passes"]),
            "n_problems_gaining_passes": len(agg["problems_gaining_passes"]),
            "problems_losing_passes": sorted(agg["problems_losing_passes"],
                                             key=lambda x: x[1] - x[2],
                                             reverse=True)[:25],
            "problems_gaining_passes": agg["problems_gaining_passes"][:25],
            "shared_timed_workloads": len(r),
            "latency_ratio_after_over_before_p50": statistics.median(r) if r else None,
            "latency_ratio_min": min(r) if r else None,
            "latency_ratio_max": max(r) if r else None,
            "n_slower_after": sum(1 for x in r if x > 1.02),
            "n_faster_after": sum(1 for x in r if x < 0.98),
        }

    payload = {
        "before": str(a.before), "after": str(a.after),
        "problems_compared": len(keys),
        "only_in_before": only_before, "only_in_after": only_after,
        "by_variant": summary,
        "n_problems_with_winner_change": len(winner_changes),
        "winner_changes": winner_changes,
        "per_problem": problems,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=1))

    print(f"compared {len(keys)} problems")
    for name, s in summary.items():
        mark = " <-- compile" if name in COMPILE_VARIANTS else ""
        # Signed as a DELTA in passes, so the sign reads the way a reader
        # expects: negative means the re-run passes fewer, which for a compile
        # variant is the whole point.
        print(f"  {name:<26} passed {s['passed_before']:>5} -> {s['passed_after']:>5} "
              f"({s['passed_after'] - s['passed_before']:+d}), "
              f"latency x{s['latency_ratio_after_over_before_p50'] or float('nan'):.3f} p50 "
              f"over {s['shared_timed_workloads']} shared{mark}")
    print(f"  winner changed on {len(winner_changes)} problems")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
