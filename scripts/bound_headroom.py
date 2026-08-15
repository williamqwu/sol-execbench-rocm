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


WHICH `T_SOL` THE RATIO IS TAKEN AGAINST (D63)
-----------------------------------------------
A manifest carries two millisecond columns and they are not the same number.
`t_sol_ms` is the legacy one: a cycle count divided by whatever reference clock
the tier that wrote it happened to use, which on MI355X is 1.8 GHz for one tier
and 2.4 GHz for the other. `t_sol_ms_published` is the bound the manifest
actually publishes -- re-derived at the minimum of the T_b measurement's own
clock bracket, which is the tightest and the only detectable end (see
`solexbench_rocm.t_sol_at`).

Measured over `artifacts/09-MI355X/manifest-v4.json`: the two differ on 3685 of
3717 scoreable workloads, over a range of 0.7481x to 1.3370x, and reading the
legacy column moves **147 workloads into a different `bound_quality` band and
214 into a different headroom band**. A headroom figure taken against a bound
nothing is scored against is not a description of this benchmark.

`published_bound_ms` below is the one place that choice is made, so the D39
report, the score distribution and the leaderboard's per-workload marking cannot
drift apart about it -- two consumers quietly disagreeing about which column a
millisecond lives in is precisely how D63 happened.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from provenance import write_artifact  # noqa: E402
from solexbench_rocm.t_sol_at import (  # noqa: E402
    MissingReferenceClock,
    bound_ms,
)

#: What `published_bound_ms` returns as its second element. Named, because
#: "which column did this number come from" has to be countable and reportable,
#: not inferable from the value.
PUBLISHED = "published"            #: t_sol_ms_published, the manifest's own bound
LEGACY_AT_STATED_CLOCK = "legacy_at_stated_clock"   #: t_sol_ms + its own f_ref_mhz
LEGACY_UNSTAMPED = "legacy_unstamped"               #: t_sol_ms, clock unstated
NO_BOUND = "no_bound"              #: nothing usable; the value is None


def published_bound_ms(w: dict) -> tuple[float | None, str]:
    """`(T_SOL_ms, basis)` for one manifest workload record.

    The preference order, and why each step is where it is:

    1. **`t_sol_ms_published`.** The bound the manifest publishes and the one a
       score is computed against. If it is there, nothing else is a candidate.
    2. **`t_sol_ms`, but only through `t_sol_at.bound_ms`**, which refuses a
       stored millisecond column that does not carry its own `f_ref_mhz`. That
       accessor exists precisely to stop a third consumer repeating D63, and
       routing the fallback through it means the legible legacy records are read
       under the same rule the tier writers agreed on.
    3. **`t_sol_ms` raw, reported as `legacy_unstamped`.** MI350X's frozen
       `manifest-v1.json` and `manifest-v1.2.json` carry `t_sol_ms_published` on
       **0 of 3717** scoreable workloads and `f_ref_mhz` on none of them either:
       they predate both fields and they are frozen, so they will never gain
       them. A hard switch at step 1, or a raise at step 2, blinds the entire
       MI350X board -- every headroom, every `bound_quality`, every band -- which
       is a bigger error than the ambiguity it was avoiding. On a part that was
       clock-locked at a single measured F_LOCK the column is unambiguous in
       fact even though the record does not say so, which is why this degrades
       rather than refuses.

    The basis comes back with the number so a caller can COUNT step 3 instead of
    taking it silently. An artifact or a board that reports how many of its rows
    are on an unstated clock is one a reader can weigh; one that does not is
    indistinguishable from a board with no legacy rows at all.

    A non-positive bound is `NO_BOUND`, not a number: `T_b / 0` is not a
    headroom, and a zero bound has been produced by rounding before (see
    `t_sol_at.t_sol_cycles_at`).
    """
    published = w.get("t_sol_ms_published")
    if published is not None and published > 0:
        return float(published), PUBLISHED
    try:
        ms = bound_ms(w)
        if ms > 0:
            return ms, LEGACY_AT_STATED_CLOCK
        return None, NO_BOUND
    except MissingReferenceClock:
        pass
    except KeyError:
        # Stamped with a clock but carrying no `t_sol_ms`: not a bounded
        # workload at all, and `bound_ms` says so in its own words.
        return None, NO_BOUND
    legacy = w.get("t_sol_ms")
    if legacy is not None and legacy > 0:
        return float(legacy), LEGACY_UNSTAMPED
    return None, NO_BOUND


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
    bases: dict[str, int] = {}

    for key, prob in man["problems"].items():
        hs, srcs, botts = [], set(), set()
        for w in prob.get("workloads", {}).values():
            t_sol, basis = published_bound_ms(w)
            t_b = w.get("t_b_ms")
            if not t_sol or not t_b:
                continue
            # Counted after the guard, so `bound_basis` sums to `workloads` and
            # describes the bounds that are actually in this distribution rather
            # than every record in the manifest.
            bases[basis] = bases.get(basis, 0) + 1
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
        "measure": ("headroom = T_b / T_SOL, per workload, with T_SOL the "
                    "manifest's PUBLISHED bound (t_sol_ms_published) wherever "
                    "it exists -- see published_bound_ms and D63"),
        "manifest": str(a.manifest.relative_to(ROOT)),
        "manifest_version": man.get("manifest_version"),
        "workloads": len(all_h),
        # Which column each bound came out of, counted rather than assumed. On
        # MI350X this is `legacy_unstamped` for all 3717 -- the frozen manifests
        # predate both `t_sol_ms_published` and `f_ref_mhz` -- and a reader is
        # entitled to know that before comparing the two parts' distributions.
        "bound_basis": dict(sorted(bases.items())),
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
    print("  bound basis      "
          + "  ".join(f"{k} {v}" for k, v in sorted(bases.items())))
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
