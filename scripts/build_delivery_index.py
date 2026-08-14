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


def stage_bound(t_sol: dict | None, key: str) -> dict:
    if not t_sol:
        return {"present": False}
    rec = (t_sol.get("problems") or {}).get(key)
    if rec is None:
        return {"present": False}
    wl = rec.get("workloads") or {}
    # the four fields t_sol_at needs to re-max the bound at a measured clock;
    # a bound that cannot be re-clocked is not usable on an unlocked part
    reclockable = sum(
        1 for w in wl.values()
        if isinstance(w, dict)
        and w.get("compute_cycles") is not None
        and w.get("dram_byte_per_sec") is not None
    )
    return {"present": True, "workloads": len(wl), "reclockable": reclockable,
            "all_reclockable": bool(wl) and reclockable == len(wl)}


def stage_tolerance(root: Path, key: str) -> dict:
    f = root / "artifacts" / "05-MI355X" / f"{key}.json"
    if not f.is_file():
        return {"present": False}
    return {"present": _loads(f) is not None}


def stage_tb(root: Path, key: str) -> dict:
    for sub in ("authoritative", "candidates"):
        f = root / "artifacts" / "06-MI355X" / sub / f"{key}.json"
        if f.is_file() and _loads(f) is not None:
            return {"present": True, "tier": sub}
    return {"present": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--benchmark-dir",
                    default=str(ROOT / "data" / "SOL-ExecBench" / "benchmark"), type=Path)
    ap.add_argument("--t-sol", default=str(ROOT / "artifacts" / "03-MI355X" / "t_sol.json"),
                    type=Path)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "delivery" / "index-40.json"),
                    type=Path)
    ap.add_argument("--markdown", type=Path, help="also write a human-readable table")
    a = ap.parse_args()

    by_cat = discover(a.benchmark_dir)
    picked, how = choose(by_cat, a.n)
    t_sol = _loads(a.t_sol) if a.t_sol.is_file() else None

    rows = []
    for cat, name in picked:
        key = f"{cat}__{name}"
        r = {
            "category": cat,
            "problem": name,
            "key": key,
            "reference": stage_reference(ROOT, key),
            "tolerance": stage_tolerance(ROOT, key),
            "bound": stage_bound(t_sol, key),
            "t_b": stage_tb(ROOT, key),
        }
        r["scoreable"] = all(r[s]["present"] for s in
                             ("reference", "tolerance", "bound", "t_b"))
        rows.append(r)

    counts = {s: sum(1 for r in rows if r[s]["present"])
              for s in ("reference", "tolerance", "bound", "t_b")}
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
