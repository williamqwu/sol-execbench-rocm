#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 09 — freeze the scoring manifest.

A SOL score is only meaningful *inside* a manifest version. The manifest is the
complete statement of what a score means: the bound it is measured against
(T_SOL), the anchor that puts S=0.5 somewhere (T_b), the tolerances a
submission must satisfy, and the exact hardware and software the two reference
numbers were produced on.

    python scripts/build_manifest.py --out artifacts/09/manifest-v1.json

Rules this script enforces rather than assumes:

* **Never edit a manifest in place.** Any stack change that moves T_b needs a
  new version. The script refuses to overwrite an existing file without
  --force, and records the git SHA it was built from.
* **Count honestly.** Every problem is either in the manifest or in
  `artifacts/deferred.json` with a reason. The totals printed here are the
  numbers that must appear in the README, the paper, and any leaderboard --
  if it is 220 and not 235, it is 220 everywhere.
* **A workload with a T_SOL but no T_b is not scoreable** and is reported as
  such rather than shipped with a guessed anchor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from provenance import write_artifact  # noqa: E402

EXPECTED = {"L1": 94, "L2": 82, "Quant": 33, "FlashInfer-Bench": 26}


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def collect_t_sol(path: Path) -> dict[str, dict]:
    """{problem: {workload_uuid: {...}}} from artifacts/03/t_sol.json."""
    doc = _load(path)
    if not doc:
        return {}
    return {k: (v.get("workloads") or {}) for k, v in doc.get("problems", {}).items()}


def combine_bounds(solar: dict, traffic: dict, tb: dict) -> tuple[dict, dict]:
    """One T_SOL per workload, from two derivations, with the source recorded.

    Both are lower bounds on the same quantity and neither dominates:

      solar_fused       accounts for the arithmetic, but only for the graph
                        SOLAR managed to extract. On 48 problems that graph is
                        missing tensors the definition itself declares, so the
                        bound is real but loose.
      declared_traffic  every declared input read once and every declared
                        output written once, over DRAM bandwidth. Accounts for
                        no arithmetic at all, but it is complete.

    The larger of two valid lower bounds is the better lower bound, so the rule
    is `max`, with one exception that is not optional: where a problem declares
    a tensor it *indexes* rather than streams -- a 131072-position KV cache --
    the declared total is above any real kernel's traffic and the "bound" would
    sit above the measured time. Those are caught by comparing against T_b and
    fall back to SOLAR's value, with the fallback recorded.
    """
    out: dict[str, dict] = {}
    stats = {"solar_fused": 0, "declared_traffic": 0, "max_of_both": 0,
             "traffic_rejected_above_t_b": 0, "solar_rejected_above_t_b": 0,
             "no_valid_bound": 0}
    for key in set(solar) | set(traffic):
        s_w, t_w = solar.get(key, {}), traffic.get(key, {})
        merged: dict[str, dict] = {}
        for u in set(s_w) | set(t_w):
            s, t = s_w.get(u) or {}, t_w.get(u) or {}
            s_cyc, t_cyc = s.get("t_sol_cycles"), t.get("t_sol_cycles")
            measured = ((tb.get(key) or {}).get(u) or {}).get("t_b_ms")
            # A candidate bound above the measured time is not a loose lower
            # bound, it is not a lower bound at all -- it would make
            # (T_b - T_SOL) negative and push scores past 1. The rule is
            # symmetric: reject any candidate that fails, take the max of what
            # survives, and if nothing survives the workload is not scoreable
            # and is counted as such rather than shipped with a bad anchor.
            if measured is not None:
                if t_cyc is not None and t.get("t_sol_ms", 0) > measured:
                    stats["traffic_rejected_above_t_b"] += 1
                    t_cyc = None
                if s_cyc is not None and s.get("t_sol_ms", 0) > measured:
                    stats["solar_rejected_above_t_b"] += 1
                    s_cyc = None
            if s_cyc is not None and t_cyc is not None:
                source = "max_of_both" if t_cyc > s_cyc else "solar_fused"
                chosen = t if t_cyc > s_cyc else s
            elif s_cyc is not None:
                source, chosen = "solar_fused", s
            elif t_cyc is not None:
                source, chosen = "declared_traffic", t
            else:
                stats["no_valid_bound"] += 1
                continue
            stats[source] += 1
            merged[u] = {**chosen, "t_sol_source": source,
                         "t_sol_cycles_solar": s_cyc,
                         "t_sol_cycles_traffic": t.get("t_sol_cycles")}
        if merged:
            out[key] = merged
    return out, stats


def collect_t_b(directory: Path) -> dict[str, dict]:
    """{problem: {workload_uuid: {variant, t_b_ms}}} from artifacts/06."""
    out: dict[str, dict] = {}
    if not directory.exists():
        return out
    for f in sorted(directory.glob("*.json")):
        doc = _load(f)
        if doc and doc.get("winner_by_workload"):
            out[doc.get("problem", f.stem)] = doc["winner_by_workload"]
    return out


def _methodology_of(directory: Path) -> str:
    """Which timing methodology produced the T_b measurements.

    Read from the artifacts rather than assumed, and a mixture is reported as
    a mixture instead of being collapsed to whichever came first.
    """
    seen = set()
    for f in sorted(directory.glob("*.json")):
        doc = _load(f) or {}
        prov = doc.get("_provenance") or {}
        m = prov.get("methodology") or (doc.get("environment") or {}).get("methodology")
        if m:
            seen.add(m)
    if not seen:
        return "hip_events"        # the harness default; see device.py
    return "+".join(sorted(seen))


def collect_tolerances(directory: Path) -> dict[str, dict]:
    """{problem: {workload_uuid: tolerance}} from artifacts/05."""
    out: dict[str, dict] = {}
    if not directory.exists():
        return out
    for f in sorted(directory.glob("*.json")):
        doc = _load(f)
        if not doc:
            continue
        per = {}
        for w in doc.get("per_workload", []):
            if w.get("tolerance"):
                per[w["workload_uuid"]] = w["tolerance"]
        if per:
            out[doc.get("problem", f.stem)] = per
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/09/manifest-v1.json")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--t-sol", default="artifacts/03/t_sol.json")
    ap.add_argument("--t-sol-traffic", default="artifacts/03/t_sol_traffic.json")
    ap.add_argument("--t-b", default="artifacts/06/authoritative")
    ap.add_argument("--tolerances", default="artifacts/05")
    ap.add_argument("--deferred", default="artifacts/deferred.json")
    ap.add_argument("--data", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest (do not do this to a "
                         "published one -- cut a new version instead)")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit(
            f"{out} already exists. A manifest is frozen once published: "
            f"scores are only comparable within a version. Cut a new version "
            f"instead, or pass --force if this one was never published."
        )

    methodology = _methodology_of(Path(a.t_b))
    t_b = collect_t_b(Path(a.t_b))
    t_sol, bound_sources = combine_bounds(
        collect_t_sol(Path(a.t_sol)),
        collect_t_sol(Path(a.t_sol_traffic)),
        t_b,
    )
    tolerances = collect_tolerances(Path(a.tolerances))
    # The ledger is `{_note, dataset_total, ..., problems: {key: reason}}`.
    # Read the mapping out of it rather than iterating the whole document --
    # `sorted(doc)` over the outer dict would list "_note" as a deferred
    # problem and inflate every count that quotes this file.
    deferred_doc = _load(Path(a.deferred)) or {}
    deferred = deferred_doc.get("problems", {})
    if not isinstance(deferred, dict):
        sys.exit(f"{a.deferred}: 'problems' must map problem key -> reason")

    data = Path(a.data)
    census = {
        f"{cat}__{p.name}": cat
        for cat in EXPECTED
        for p in sorted((data / cat).glob("*"))
        if (p / "definition.json").exists()
    }

    problems: dict[str, dict] = {}
    stats = {"scoreable_workloads": 0, "workloads_missing_t_sol": 0,
             "workloads_missing_t_b": 0, "workloads_missing_tolerance": 0}

    for key, category in sorted(census.items()):
        sol = t_sol.get(key, {})
        tb = t_b.get(key, {})
        tol = tolerances.get(key, {})
        uuids = sorted(set(sol) | set(tb) | set(tol))
        entries = {}
        for u in uuids:
            s, b = sol.get(u, {}), tb.get(u, {})
            has_sol = "t_sol_cycles" in s
            has_tb = "t_b_ms" in b
            if not has_sol:
                stats["workloads_missing_t_sol"] += 1
            if not has_tb:
                stats["workloads_missing_t_b"] += 1
            if u not in tol:
                stats["workloads_missing_tolerance"] += 1
            if has_sol and has_tb:
                stats["scoreable_workloads"] += 1
            entries[u] = {
                # Cycles first: it is the F_LOCK-invariant figure, so a future
                # re-lock rescales the ms column by one division instead of
                # invalidating the manifest's analytic half.
                "t_sol_cycles": s.get("t_sol_cycles"),
                "t_sol_ms": s.get("t_sol_ms"),
                # Which derivation produced the bound, and what the other one
                # said. Two lower bounds on the same quantity, neither of them
                # dominating -- a consumer that cares can filter on this.
                "t_sol_source": s.get("t_sol_source"),
                "t_sol_cycles_solar": s.get("t_sol_cycles_solar"),
                "t_sol_cycles_traffic": s.get("t_sol_cycles_traffic"),
                "sol_bottleneck": s.get("bottleneck"),
                "t_b_ms": b.get("t_b_ms"),
                # "Optimized PyTorch" is not reproducible; a named variant is.
                "t_b_variant": b.get("variant"),
                "tolerance": tol.get(u),
                "scoreable": has_sol and has_tb,
            }
        problems[key] = {
            "category": category,
            "n_workloads": len(entries),
            "n_scoreable": sum(1 for e in entries.values() if e["scoreable"]),
            "workloads": entries,
            "deferred": deferred.get(key),
        }

    scoreable_problems = [k for k, v in problems.items() if v["n_scoreable"]]
    payload = {
        "manifest_version": a.version,
        # Stated at the top level, not buried in provenance: a manifest built
        # from hip_events traces and one built from rocprof traces are not
        # comparable, and the whole point of recording the methodology per
        # trace is lost if the manifest that aggregates them does not say
        # which one it aggregated.
        "methodology": methodology,
        "score_formula": "S(T_k) = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))",
        "problem_set": {
            "total_in_dataset": len(census),
            "expected_by_category": EXPECTED,
            "scoreable_problems": len(scoreable_problems),
            "deferred_problems": sorted(deferred),
            # Stated once, here, so every other document can quote one number
            # rather than each computing its own and drifting.
            "headline_count": len(scoreable_problems),
        },
        "stats": stats,
        "bound_sources": bound_sources,
        "problems": problems,
    }
    write_artifact(out, f"09-manifest-{a.version}", payload)

    print(f"manifest {a.version} -> {out}")
    print(f"  problems scoreable   {len(scoreable_problems)}/{len(census)}")
    print(f"  workloads scoreable  {stats['scoreable_workloads']}")
    for k in ("workloads_missing_t_sol", "workloads_missing_t_b",
              "workloads_missing_tolerance"):
        print(f"  {k:<28} {stats[k]}")
    if len(scoreable_problems) < len(census):
        missing = sorted(set(census) - set(scoreable_problems) - set(deferred))
        print(f"\n{len(missing)} problems are neither scoreable nor recorded in "
              f"{a.deferred}. Each is a gap without a decision behind it:")
        for m in missing[:20]:
            print(f"  {m}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


if __name__ == "__main__":
    main()
