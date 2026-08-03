#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Coverage ledger — is every problem in the dataset accounted for?

The scope of this port is **all 235 problems**. The realistic way that stops
being true is not a decision, it is an omission: a `--category` flag missing an
entry, a sweep that died partway and was marked done, a crashed worker whose
problem was never retried. Each of those looks like success and quietly shrinks
the benchmark.

This compares the dataset census against an artifact directory and names every
problem that is missing. Run it after every sweep.

    python scripts/check_coverage.py --artifacts artifacts/02/references
    python scripts/check_coverage.py --artifacts artifacts/05/workloads --pattern 'workload.jsonl'

Exit 0 only when coverage is complete, or when every gap is listed in
`artifacts/deferred.json` with a reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED = {"L1": 94, "L2": 82, "Quant": 33, "FlashInfer-Bench": 26}
TOTAL = sum(EXPECTED.values())   # 235


def census(data: Path) -> dict[str, list[str]]:
    """Every problem in the dataset, by category."""
    out: dict[str, list[str]] = {}
    for cat in EXPECTED:
        cat_dir = data / cat
        out[cat] = sorted(
            p.name for p in cat_dir.iterdir()
            if (p / "definition.json").exists()
        ) if cat_dir.is_dir() else []
    return out


def covered(art: Path, pattern: str | None) -> set[str]:
    """Problems present in the artifact dir, as 'Category__problem' keys."""
    if not art.exists():
        return set()
    if pattern:
        # nested layout: <artifacts>/<Category>/<problem>/<pattern>
        return {f"{p.parent.parent.name}__{p.parent.name}"
                for p in art.rglob(pattern)}
    # flat layout: <artifacts>/<Category>__<problem>.json
    return {p.stem for p in art.glob("*.json")}


def deferrals(root: Path) -> dict[str, str]:
    f = root / "artifacts" / "deferred.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text()).get("problems", {})
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--pattern", help="nested-layout filename, e.g. workload.jsonl")
    ap.add_argument("--json-out")
    a = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    data = Path(a.data)
    if not data.exists():
        sys.exit(f"dataset not found at {data} — cannot verify coverage")

    cen = census(data)
    have = covered(Path(a.artifacts), a.pattern)
    deferred = deferrals(root)

    print(f"\nCoverage — {a.artifacts}\n")
    missing_all: list[str] = []
    for cat, problems in cen.items():
        exp = EXPECTED[cat]
        if len(problems) != exp:
            print(f"  {cat:<18} DATASET MISMATCH: found {len(problems)}, "
                  f"expected {exp}")
        miss = [p for p in problems if f"{cat}__{p}" not in have]
        miss_undeferred = [p for p in miss if f"{cat}__{p}" not in deferred]
        missing_all.extend(f"{cat}__{p}" for p in miss_undeferred)
        got = len(problems) - len(miss)
        flag = "" if not miss_undeferred else f"  <-- {len(miss_undeferred)} MISSING"
        print(f"  {cat:<18} {got:>3}/{len(problems):<3} covered"
              f"{'' if not miss else f'  ({len(miss) - len(miss_undeferred)} deferred)'}{flag}")

    total_problems = sum(len(v) for v in cen.values())
    total_covered = total_problems - len(missing_all) - len(
        [k for k in deferred if k in {f"{c}__{p}" for c, ps in cen.items() for p in ps}])
    print(f"\n  TOTAL              {total_problems - len(missing_all)}/{total_problems} "
          f"(scope is all {TOTAL})")

    if missing_all:
        print(f"\n  {len(missing_all)} problem(s) neither covered nor deferred:")
        for k in missing_all[:25]:
            print(f"    {k}")
        if len(missing_all) > 25:
            print(f"    ... and {len(missing_all) - 25} more")
        print("\n  Either run them, or record each in artifacts/deferred.json as")
        print('    {"problems": {"<Category>__<problem>": "<reason>"}}')
        print("  A gap with a stated reason is a decision. A gap without one is a bug.")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(
            {"expected_total": TOTAL, "dataset_total": total_problems,
             "covered": total_problems - len(missing_all),
             "missing": missing_all, "deferred": deferred}, indent=2))

    sys.exit(1 if missing_all else 0)


if __name__ == "__main__":
    main()
