#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Index a delivery slice: which problems are ready, and how far each one got.

    python scripts/build_delivery_index.py --n 40 --out artifacts/delivery/index-40.json

The benchmark ships as a pipeline -- reference correctness, tolerance
calibration, a Speed-of-Light bound, a T_b anchor -- and a problem is only
scoreable when every stage has produced an artifact for it. This script reads
the artifact tree and reports, per problem, which stages are present. It does
not run anything and it does not infer: a stage is present when its file is on
disk and parses, and absent otherwise.

The slice is chosen by a stated rule rather than by whatever happened to finish
first, because "the 40 that are done" is not reproducible and quietly selects
for the problems that are cheap to measure. The rule is: proportional by
category, first N of each in sorted order. It is a pure function of --n and the
dataset, so the same 40 come back on any node, before any of them are measured.

Stages are reported, never averaged into a single percentage. A problem with a
bound and no anchor is not "75% done"; it is unscoreable, and the index says
which stage is missing so the next run knows what to do.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

CATEGORIES = ["L1", "L2", "Quant", "FlashInfer-Bench"]


def discover(benchmark_dir: Path) -> dict[str, list[str]]:
    """Every problem on disk, by category, sorted. The dataset is the census."""
    out: dict[str, list[str]] = {}
    for cat in CATEGORIES:
        d = benchmark_dir / cat
        if not d.is_dir():
            out[cat] = []
            continue
        out[cat] = sorted(p.name for p in d.iterdir()
                          if p.is_dir() and (p / "definition.json").is_file())
    return out


def choose(by_cat: dict[str, list[str]], n: int) -> tuple[list[tuple[str, str]], dict]:
    """Proportional by category, first N of each, sorted.

    Largest-remainder, so the parts sum to exactly n rather than to n plus or
    minus a rounding error -- an off-by-one here would be a silently short
    delivery, which is the failure mode CLAUDE.md section 0 is about.
    """
    total = sum(len(v) for v in by_cat.values())
    if total == 0:
        return [], {"error": "no problems found"}
    exact = {c: n * len(v) / total for c, v in by_cat.items()}
    floor = {c: int(x) for c, x in exact.items()}
    short = n - sum(floor.values())
    # hand the leftover seats to the largest fractional parts, ties by category
    # order, so the choice is deterministic rather than dict-order dependent
    order = sorted(by_cat, key=lambda c: (-(exact[c] - floor[c]), CATEGORIES.index(c)))
    for c in order[:short]:
        floor[c] += 1
    picked: list[tuple[str, str]] = []
    for c in CATEGORIES:
        picked.extend((c, p) for p in by_cat.get(c, [])[: floor[c]])
    return picked, {"per_category": floor,
                    "available_per_category": {c: len(v) for c, v in by_cat.items()},
                    "rule": "proportional by category, first N sorted, "
                            "largest-remainder apportionment"}


def clock_span(root: Path, key: str) -> float | None:
    """Widest before/after clock ratio seen in any of this problem's windows.

    A proxy for the width of the T_SOL interval, and a conservative one: T_SOL's
    compute term goes as 1/f while its memory term is clock-invariant, so the
    bound's relative width is at most this span and is exactly zero for a
    purely memory-bound workload. Using the widest window rather than the median
    is deliberate -- the question a precision filter answers is "how badly is
    this problem's bound known at worst", not "on average".

    Returns None when nothing has bracketed the problem yet, which is different
    from a span of zero and must not be conflated with it.
    """
    lo = hi = None
    for sub in ("authoritative", "authoritative-40", "candidates"):
        d = _loads(root / "artifacts" / "06-MI355X" / sub / f"{key}.json")
        if not d:
            continue
        for v in (d.get("variants") or {}).values():
            for cb in (v.get("clock_bracket_by_workload") or {}).values():
                if not isinstance(cb, dict):
                    continue
                for f in (cb.get("clock_before_mhz"), cb.get("clock_after_mhz")):
                    if f:
                        lo = f if lo is None else min(lo, f)
                        hi = f if hi is None else max(hi, f)
    return None if not lo else hi / lo - 1.0


def _loads(p: Path):
    """A file counts as present only if it parses. A crash stub is not a result."""
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def stage_reference(root: Path, key: str) -> dict:
    f = root / "artifacts" / "02-MI355X" / "references" / f"{key}.json"
    if not f.is_file():
        return {"present": False}
    d = _loads(f)
    if d is None:
        return {"present": False, "note": "on disk but does not parse"}
    # The runner already reduces per_workload into these; recomputing from
    # per_workload would be a second, divergeable source of the same number.
    total = d.get("workloads")
    passed = d.get("passed")
    if not isinstance(total, int) or not isinstance(passed, int):
        return {"present": False, "note": "parses but carries no workload counts"}
    return {"present": True, "workloads": total, "passed": passed,
            "all_passed": bool(d.get("all_passed"))}


def _reclockable(w) -> bool:
    """Can this workload's bound be evaluated at a clock other than the one it
    was written at? Needs the terms that scale differently: a cycle count for
    the compute side and a bytes/second for the memory side."""
    return (isinstance(w, dict)
            and w.get("compute_cycles") is not None
            and w.get("dram_byte_per_sec") is not None)


def stage_bound(t_sol: dict | None, key: str,
                traffic: dict | None = None) -> dict:
    """A workload is bounded if EITHER tier bounds it.

    T_SOL comes from two derivations -- SOLAR's roofline over the traced graph,
    and the traffic the definition declares over DRAM bandwidth -- and the
    manifest takes the max of the two that survive checking against the
    measurement. Reading only the SOLAR tier here reported 8 of the first 40 as
    unbounded when the traffic tier covers them; SOLAR's tracing ceiling is a
    property of SOLAR, not of the problem.
    """
    tiers = {}
    for name, src in (("solar", t_sol), ("traffic", traffic)):
        rec = ((src or {}).get("problems") or {}).get(key)
        if rec is not None:
            tiers[name] = rec.get("workloads") or {}
    if not tiers:
        return {"present": False}

    ids = set().union(*(set(w) for w in tiers.values()))
    reclockable = sum(
        1 for i in ids
        if any(_reclockable(w.get(i)) for w in tiers.values())
    )
    return {"present": True, "workloads": len(ids),
            "reclockable": reclockable,
            "all_reclockable": bool(ids) and reclockable == len(ids),
            "tiers": sorted(tiers)}


def stage_tolerance(root: Path, key: str) -> dict:
    f = root / "artifacts" / "05-MI355X" / f"{key}.json"
    if not f.is_file():
        return {"present": False}
    return {"present": _loads(f) is not None}


def stage_tb(root: Path, key: str) -> dict:
    for sub in ("authoritative", "authoritative-40", "candidates"):
        f = root / "artifacts" / "06-MI355X" / sub / f"{key}.json"
        if f.is_file() and _loads(f) is not None:
            tier = "authoritative" if sub.startswith("authoritative") else sub
            return {"present": True, "tier": tier, "source": sub}
    return {"present": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--benchmark-dir",
                    default=str(ROOT / "data" / "SOL-ExecBench" / "benchmark"), type=Path)
    ap.add_argument("--t-sol", default=str(ROOT / "artifacts" / "03-MI355X" / "t_sol.json"),
                    type=Path)
    ap.add_argument("--t-sol-traffic",
                    default=str(ROOT / "artifacts" / "03-MI355X" / "t_sol_traffic.json"),
                    type=Path,
                    help="the declared-traffic tier. T_SOL is the max of two "
                         "derivations; reading only SOLAR under-reports coverage "
                         "wherever its tracing failed.")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "delivery" / "index-40.json"),
                    type=Path)
    ap.add_argument("--markdown", type=Path, help="also write a human-readable table")
    ap.add_argument("--max-clock-span", type=float,
                    help="keep only problems whose widest in-window clock span is "
                         "at most this (0.05 = 5%%). On an unlocked part the T_SOL "
                         "interval is at most this wide, so it selects for "
                         "problems whose bound is precisely known. Excluded "
                         "problems are listed with their measured span -- the "
                         "subset must never look like it selected itself.")
    a = ap.parse_args()

    by_cat = discover(a.benchmark_dir)

    # Precision filter. This is the ONE criterion allowed to shrink the pool, and
    # it is a measured property with a stated threshold, not "drop what failed".
    # The difference matters: excluding problems because they did not work makes
    # the result "the N that happened to be measurable", and every rate quoted
    # over it is then biased by the same effect that caused the exclusion. A
    # threshold on a published quantity is reproducible, arguable, and visible.
    excluded_by_span: dict[str, float | None] = {}
    if a.max_clock_span is not None:
        for cat, names in by_cat.items():
            keep = []
            for p in names:
                key = f"{cat}__{p}"
                span = clock_span(ROOT, key)
                if span is not None and span <= a.max_clock_span:
                    keep.append(p)
                else:
                    excluded_by_span[key] = span
            by_cat[cat] = keep
    picked, how = choose(by_cat, a.n)
    t_sol = _loads(a.t_sol) if a.t_sol.is_file() else None
    traffic = _loads(a.t_sol_traffic) if a.t_sol_traffic.is_file() else None

    rows = []
    for cat, name in picked:
        key = f"{cat}__{name}"
        r = {
            "category": cat,
            "problem": name,
            "key": key,
            "reference": stage_reference(ROOT, key),
            "tolerance": stage_tolerance(ROOT, key),
            "bound": stage_bound(t_sol, key, traffic),
            "t_b": stage_tb(ROOT, key),
        }
        # "Scoreable" is a claim about whether S can actually be computed, not
        # about whether four files exist. Two of the four stages can be present
        # and still not support a score, and an earlier version of this script
        # counted them as if they did -- which is the same defect as a gate that
        # passes over an empty list, wearing a stronger name than it earns.
        #
        #   t_b must be AUTHORITATIVE. The candidates tier selects which variant
        #   becomes the anchor; it is not the re-timed anchor. Counting it would
        #   report a problem as scoreable before the pass that anchors it ran.
        #
        #   the bound must be RE-CLOCKABLE. On the unlocked basis T_SOL is
        #   evaluated at the measurement's own bracket clock, which needs
        #   compute_cycles and dram_byte_per_sec per workload. A bound carrying
        #   only a millisecond figure is a bound at some other clock.
        reasons = []
        if not r["reference"]["present"]:
            reasons.append("no reference")
        if not r["tolerance"]["present"]:
            reasons.append("no tolerance")
        if not r["bound"]["present"]:
            reasons.append("no bound")
        elif not r["bound"].get("all_reclockable"):
            reasons.append("bound not re-clockable")
        if not r["t_b"]["present"]:
            reasons.append("no T_b")
        elif r["t_b"].get("tier") != "authoritative":
            reasons.append("T_b is candidate-tier, not authoritative")
        r["scoreable"] = not reasons
        r["not_scoreable_because"] = reasons
        rows.append(r)

    counts = {s: sum(1 for r in rows if r[s]["present"])
              for s in ("reference", "tolerance", "bound", "t_b")}
    counts["t_b_authoritative"] = sum(
        1 for r in rows if r["t_b"].get("tier") == "authoritative")
    counts["bound_reclockable"] = sum(
        1 for r in rows if r["bound"].get("all_reclockable"))
    counts["scoreable"] = sum(1 for r in rows if r["scoreable"])

    try:
        from provenance import stamp
        prov = stamp("delivery-index")
    except Exception as e:  # noqa: BLE001
        # never fabricate provenance; record that it could not be taken
        prov = {"_provenance_error": repr(e)}

    doc = {**prov, "n_requested": a.n, "n_selected": len(rows),
           "selection": how, "stage_counts": counts,
           "t_sol_source": str(a.t_sol) if t_sol else None,
           # Every problem the precision filter removed, with the span that
           # removed it. Without this the subset would look like it selected
           # itself, and a reader could not tell a deliberate precision floor
           # from a quiet exclusion of everything inconvenient.
           "precision_filter": (
               None if a.max_clock_span is None else {
                   "max_clock_span": a.max_clock_span,
                   "rationale":
                       "T_SOL's compute term goes as 1/f, so the in-window clock "
                       "span bounds the width of the T_SOL interval. The anchor "
                       "property is 0.5 +- 0.03, so a bound known to worse than "
                       "about that is not usable for ranking. This threshold is "
                       "the measured quantity, not a judgement about the problem.",
                   "n_excluded": len(excluded_by_span),
                   "excluded": dict(sorted(
                       excluded_by_span.items(),
                       key=lambda kv: (kv[1] is None, -(kv[1] or 0)))),
               }),
           "problems": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=2, default=str))

    print(f"selected {len(rows)} of {a.n} requested; {how.get('per_category')}")
    for s, c in counts.items():
        print(f"  {s:<12} {c}/{len(rows)}")
    print(f"wrote {a.out}")

    if a.markdown:
        tick = lambda b: "yes" if b else "-"  # noqa: E731
        lines = [f"# Delivery index — {len(rows)} problems (MI355X)", "",
                 f"Selection: {how.get('rule')}", "",
                 "| # | category | problem | reference | tolerance | bound | T_b | scoreable |",
                 "|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(rows, 1):
            lines.append(
                f"| {i} | {r['category']} | `{r['problem']}` | "
                f"{tick(r['reference']['present'])} | {tick(r['tolerance']['present'])} | "
                f"{tick(r['bound']['present'])} | {tick(r['t_b']['present'])} | "
                f"{tick(r['scoreable'])} |")
        lines += ["", "Counts: " + ", ".join(f"{s} {c}/{len(rows)}"
                                             for s, c in counts.items())]
        a.markdown.parent.mkdir(parents=True, exist_ok=True)
        a.markdown.write_text("\n".join(lines) + "\n")
        print(f"wrote {a.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
