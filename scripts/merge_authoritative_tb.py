#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge authoritative T_b trees measured on different cards and nodes.

    python scripts/merge_authoritative_tb.py \
        --in artifacts/06-MI355X/authoritative \
        --in artifacts/06-MI355X/authoritative-g05 \
        --in artifacts/06-MI355X/authoritative-40 \
        --out artifacts/06-MI355X/authoritative-merged

The authoritative pass runs 8-way card-pinned on three nodes, so the same
problem may have been anchored more than once on different silicon. Something
has to choose, and the choice must be a stated rule rather than whatever
``cp -n`` happened to do first -- the previous merged tree was assembled by
hand and no artifact records how.

**The rule, per problem:** take the artifact that anchored the most workloads;
break ties by the lower median anchor time (the faster card measured a cleaner
window); break remaining ties by sorted source order, so the merge is
deterministic.

**What is NOT done here, deliberately.** Workloads are not cherry-picked across
sources into a synthetic best-of record. A T_b set is only coherent as a set:
its workloads were measured in one session on one card, and mixing them would
produce an anchor no card ever exhibited. Merging is at problem granularity.

**What this tree is for.** It is the *published* anchor -- the T_b that fixes
S=0.5 in the manifest. It is NOT the anchor a scoring run should pair against:
decision 4 (STATE.md 4.4) requires T_b and T_k to be re-timed back to back on
the same card, so a scoring run uses the single-card tree matching the card it
runs on and enforces ``card_identity``. Merging across cards here is correct
for publication and would be wrong for pairing; the distinction is why every
record keeps the ``card_identity`` it was measured under.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import stamp  # noqa: E402


def anchored(doc: dict) -> int:
    return len(doc.get("winner_by_workload") or {})


def median_tb(doc: dict) -> float:
    """Median winning T_b, or +inf when the record anchors nothing."""
    times = []
    for w in (doc.get("winner_by_workload") or {}).values():
        v = w.get("t_b_ms") if isinstance(w, dict) else None
        if isinstance(v, (int, float)):
            times.append(float(v))
    return statistics.median(times) if times else float("inf")


def card_of(doc: dict) -> str:
    ci = doc.get("card_identity") or {}
    if isinstance(ci, dict):
        return f"{ci.get('hostname', '?')}:{ci.get('bdf') or ci.get('drm_card') or '?'}"
    return str(ci)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="sources", action="append", required=True,
                    type=Path, help="repeatable, in preference order for ties")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # problem -> list of (source_rank, path, doc)
    by_problem: dict[str, list] = {}
    for rank, src in enumerate(args.sources):
        if not src.is_dir():
            print(f"  missing source, skipped: {src}")
            continue
        for p in sorted(src.glob("*.json")):
            try:
                doc = json.loads(p.read_text())
            except Exception as e:  # noqa: BLE001
                print(f"  unreadable, skipped: {p} ({e})")
                continue
            by_problem.setdefault(p.stem, []).append((rank, p, doc))

    chosen: dict[str, tuple] = {}
    contested = 0
    for key, cands in sorted(by_problem.items()):
        if len(cands) > 1:
            contested += 1
        # most anchored, then lowest median T_b, then source order
        rank, path, doc = min(
            cands, key=lambda c: (-anchored(c[2]), median_tb(c[2]), c[0]))
        chosen[key] = (path, doc, cands)

    out = args.out
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    report = {
        **stamp("06-merge-authoritative"),
        "sources": [str(s) for s in args.sources],
        "rule": "max anchored workloads; tie -> lower median T_b; tie -> source order",
        "problems": len(chosen),
        "problems_with_more_than_one_source": contested,
        "workloads_anchored": sum(anchored(d) for _, d, _ in chosen.values()),
        "by_source": dict(Counter(str(p.parent) for p, _, _ in chosen.values())),
        "by_card": dict(Counter(card_of(d) for _, d, _ in chosen.values())),
        # Every problem where the losing source anchored a different count --
        # visible, because a large gap means one card had a much worse window
        # and that is a finding, not a merge detail.
        "displaced": sorted(
            [
                {
                    "problem": k,
                    "chosen": str(p.parent), "chosen_anchored": anchored(d),
                    "others": [
                        {"source": str(cp.parent), "anchored": anchored(cd)}
                        for _, cp, cd in cands if cp != p
                    ],
                }
                for k, (p, d, cands) in chosen.items() if len(cands) > 1
            ],
            key=lambda r: -max((o["anchored"] for o in r["others"]), default=0),
        )[:60],
    }

    for key, (path, _doc, _c) in chosen.items():
        if not args.dry_run:
            shutil.copy2(path, out / f"{key}.json")

    print(json.dumps({k: v for k, v in report.items() if k != "displaced"},
                     indent=2, default=str))
    if not args.dry_run:
        (out / "_merge-report.json").write_text(
            json.dumps(report, indent=2, default=str))
        print(f"\nwrote {len(chosen)} problems -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
