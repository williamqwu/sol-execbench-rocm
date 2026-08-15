#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure a part's T_b RE-TIMING PRECISION from two independent campaigns.

`verify_anchor._classify_headroom` exempts a workload from the anchor check when
its T_b/T_SOL headroom is too small for the anchor property to be adjudicated at
the precision the node achieves:

    h_min = eps * (0.5 + tol) / (2 * tol) = eps * 53/6   at tol = 3%

`eps` is estimated inside the anchor run itself, from its own well-conditioned
workloads. That makes the exemption self-widening: a noisier run raises eps,
raises h_min, exempts more, and reports a higher pass rate.
`verify_artifacts.MAX_H_MIN` is the ceiling that closes that loop -- but a
ceiling is only meaningful if it is a MEASURED property OF THE PART, and it must
come from somewhere other than the anchor run it is judging.

This script provides that independent source. Task 06 timed the T_b candidates
twice on MI355X -- `authoritative/` and `authoritative-repro/`, different GPUs,
different times, same code and same clock basis -- so every workload/variant
present in both is a genuine repeat measurement of the same quantity. Their
paired ratio is exactly the ``|t_k/t_b - 1|`` that the anchor run estimates,
measured 5126 times without reference to any manifest, score, or bound.

The gate draws 20 problems, so the ceiling has to bound the 20-PROBLEM sampling
distribution of the median, not the population median: it is bootstrapped by
resampling whole problems (the errors are strongly clustered by problem).

    python scripts/derive_retime_precision.py --part MI355X

Writes artifacts/06-<part>/retime-precision.json. Pure CPU, no GPU, no timing.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import write_artifact  # noqa: E402

WELL_CONDITIONED = 0.25   # matches verify_anchor._WELL_CONDITIONED
TOL = 0.03


def _load_campaign(d: Path) -> dict:
    """(problem, variant, workload_uuid) -> latency_ms, over one candidate sweep."""
    out: dict[tuple[str, str, str], float] = {}
    for p in sorted(glob.glob(str(d / "*.json"))):
        rec = json.loads(Path(p).read_text())
        if not rec.get("ok"):
            continue
        for variant, v in (rec.get("variants") or {}).items():
            for uuid, t in (v.get("latency_ms_by_workload") or {}).items():
                if t:
                    out[(rec["problem"], variant, uuid)] = t
    return out


def _h_min(eps: float, tol: float = TOL) -> float:
    return eps / (2.0 - 1.0 / (0.5 + tol))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="MI355X")
    ap.add_argument("--campaign-a", default="authoritative")
    ap.add_argument("--campaign-b", default="authoritative-repro")
    ap.add_argument("--manifest", default=None,
                    help="manifest whose t_b_variant / headroom selects the "
                         "comparable subset (default artifacts/09-<part>/manifest-v2.json)")
    ap.add_argument("--sample", type=int, default=20,
                    help="problems the anchor gate draws; the bootstrap size")
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = ROOT / "artifacts" / f"06-{a.part}"
    A, B = _load_campaign(base / a.campaign_a), _load_campaign(base / a.campaign_b)
    if not A or not B:
        print(f"missing campaign: {a.campaign_a}={len(A)} {a.campaign_b}={len(B)}")
        return 2

    man_path = Path(a.manifest) if a.manifest else (
        ROOT / "artifacts" / f"09-{a.part}" / "manifest-v2.json")
    problems = json.loads(man_path.read_text())["problems"]

    all_pairs: list[float] = []
    by_problem: dict[str, list[float]] = defaultdict(list)
    for key in sorted(set(A) & set(B)):
        problem, variant, uuid = key
        err = abs(B[key] / A[key] - 1.0)
        all_pairs.append(err)
        # The anchor run only ever re-times the manifest's winning variant, and
        # only ever calibrates eps on well-conditioned workloads. Restrict to the
        # same subset or the two numbers are not comparable.
        w = (problems.get(problem) or {}).get("workloads", {}).get(uuid)
        if not w or not w.get("scoreable") or w.get("t_b_variant") != variant:
            continue
        t_b, t_sol = w.get("t_b_ms"), w.get("t_sol_ms")
        if not t_b or t_sol is None:
            continue
        if (t_b - t_sol) / t_b >= WELL_CONDITIONED:
            by_problem[problem].append(err)

    comparable = [e for v in by_problem.values() for e in v]
    if not comparable:
        print("no comparable pairs")
        return 2

    import random
    rng = random.Random(a.seed)
    keys = sorted(by_problem)
    draws = []
    for _ in range(a.bootstrap):
        picked = [keys[rng.randrange(len(keys))] for _ in range(min(a.sample, len(keys)))]
        vals = [e for k in picked for e in by_problem[k]]
        if vals:
            draws.append(st.median(vals))
    draws.sort()

    def q(p: float) -> float:
        return draws[min(len(draws) - 1, int(p * len(draws)))]

    eps_pop = st.median(comparable)
    payload = {
        "part": a.part,
        "method": (
            "paired re-measurement: every (problem, variant, workload) timed in "
            "both candidate campaigns is one repeat of the same quantity; "
            "eps = median |t_B/t_A - 1|. Independent of any anchor run, manifest "
            "t_b, score or bound."
        ),
        "campaigns": {
            "a": f"artifacts/06-{a.part}/{a.campaign_a}",
            "b": f"artifacts/06-{a.part}/{a.campaign_b}",
        },
        "manifest_for_subset": str(man_path.relative_to(ROOT)),
        "tolerance": TOL,
        "h_min_formula": "eps * (0.5 + tol) / (2 * tol)",
        "all_pairs": {
            "n": len(all_pairs),
            "eps_median": st.median(all_pairs),
            "h_min": _h_min(st.median(all_pairs)),
        },
        "comparable_subset": {
            "n": len(comparable),
            "n_problems": len(keys),
            "criterion": (f"manifest winning variant, headroom >= "
                          f"{WELL_CONDITIONED:.0%}"),
            "eps_median": eps_pop,
            "h_min": _h_min(eps_pop),
        },
        "sampling_distribution": {
            "sample_problems": a.sample,
            "resamples": len(draws),
            "note": ("whole problems are resampled, not workloads: the re-timing "
                     "error is clustered by problem, so resampling workloads "
                     "would understate the spread the gate actually sees"),
            "h_min_percentiles": {
                f"p{int(p*100)}": _h_min(q(p))
                for p in (0.5, 0.75, 0.9, 0.95, 0.99)
            },
            "h_min_max": _h_min(draws[-1]),
        },
    }
    out = a.out or str(base / "retime-precision.json")
    write_artifact(out, "06-retime-precision", payload)

    cs = payload["comparable_subset"]
    sd = payload["sampling_distribution"]["h_min_percentiles"]
    print(f"{a.part}: {payload['all_pairs']['n']} paired re-measurements, "
          f"{cs['n']} comparable over {cs['n_problems']} problems")
    print(f"  eps (median |t_B/t_A - 1|) = {cs['eps_median']:.3%}"
          f"   -> h_min = {cs['h_min']:.2%}")
    print("  h_min over %d-problem draws: " % a.sample
          + "  ".join(f"{k} {v:.2%}" for k, v in sd.items()))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
