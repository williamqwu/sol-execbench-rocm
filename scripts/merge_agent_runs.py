#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fold a follow-up run's problems into the run it completes.

    python scripts/merge_agent_runs.py --base artifacts/10/<run> --fill artifacts/10/<run>-fill

A sweep that covered 189 of 220 problems and a later sweep of the remaining 31
are one submission, not two: same harness, same model, same cap, same node. The
board should say 220, and it can only do that if the artifacts are one run.

**The fill wins, completely, for every problem it touched.** Not merged
per-workload, not "best of the two" -- the later attempt replaces the earlier
one wherever it exists. Keeping whichever attempt scored higher would make the
submission a maximum over re-runs rather than a measurement of the harness, and
that is a different and much more flattering thing to publish. So this deletes
the base's `retimed/`, `kernels/` and `trajectory/` for those problems before
copying, rather than letting a stale artifact survive under a new session.

Nothing is scored here. After this, re-run:

    python scripts/agent_score.py --run <base> --gpu 0 --iterations 50 \\
        --warmup 10 --reuse-retimed        # re-times ONLY the deleted ones
    python scripts/import_fleet_depth.py --run <base>
    leaderboard/ingest.py

`--reuse-retimed` is what keeps this cheap: the problems the base already
measured keep their GPU-0 numbers untouched, and only the ones this script
cleared go back on the device.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=lambda p: Path(p).resolve())
    ap.add_argument("--fill", required=True, type=lambda p: Path(p).resolve())
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    base_json = a.base / "run.json"
    base = json.loads(base_json.read_text())
    fill = json.loads((a.fill / "run.json").read_text())

    fill_sessions = fill.get("sessions") or {}
    if not fill_sessions:
        print(f"{a.fill}: no sessions; nothing to merge")
        return 1

    overlap = sorted(set(base.get("sessions") or {}) & set(fill_sessions))
    fresh = sorted(set(fill_sessions) - set(base.get("sessions") or {}))
    print(f"{len(fill_sessions)} problems in the fill: "
          f"{len(fresh)} new, {len(overlap)} replacing an earlier attempt")

    # Clear the base's artifacts for every problem the fill touched, so nothing
    # from the superseded attempt can survive into the merged run. `retimed` is
    # the important one: leave it and `--reuse-retimed` will happily re-derive
    # the OLD kernel's score against the NEW session.
    removed = {"retimed": 0, "kernels": 0, "trajectory": 0, "transcripts": 0}
    for key in overlap:
        for sub, name in (("retimed", f"{key}.json"), ("kernels", f"{key}.py"),
                          ("transcripts", f"{key}.jsonl")):
            f = a.base / sub / name
            if f.exists():
                if not a.dry_run:
                    f.unlink()
                removed[sub] += 1
        d = a.base / "trajectory" / key
        if d.is_dir():
            if not a.dry_run:
                shutil.rmtree(d)
            removed["trajectory"] += 1

    base.setdefault("sessions", {}).update(fill_sessions)
    base["n_problems"] = len(base["sessions"])
    # Where each problem's session came from, so the merge is auditable from
    # the artifact rather than from this script's existence.
    merged = base.setdefault("merged_from", [])
    merged.append({"run_id": fill.get("run_id"), "problems": sorted(fill_sessions),
                   "replaced": overlap})
    gpus = sorted({g for s in base["sessions"].values()
                   for g in (s.get("gpu") or [])})
    base["gpus_used_by_agents"] = gpus

    if not a.dry_run:
        base_json.write_text(json.dumps(base, indent=1))
    print(f"cleared from the base: {removed}")
    print(f"{'would write' if a.dry_run else 'wrote'} {base_json}: "
          f"{base['n_problems']} sessions, agent GPUs {gpus}")
    print("\nnext:")
    print(f"  python3 scripts/agent_score.py --run {a.base} --gpu 0 "
          f"--iterations 50 --warmup 10 --reuse-retimed")
    print(f"  python3 scripts/import_fleet_depth.py --run {a.base}")
    print("  leaderboard/.venv/bin/python leaderboard/ingest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
