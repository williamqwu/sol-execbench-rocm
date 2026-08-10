#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Watch the authoritative GPU and record anything that lands on it.

    python scripts/guard_authoritative_gpu.py --owner-pid 350088 \
        --out artifacts/11/gpu0-guard.json --interval 20

**Why this is not redundant with the scheduler's reservation.** `TODO.md` D29:
`dash-overlay`'s J2 sweep placed 34 jobs on GPU 0 *while holding a scheduler
reservation on it*. The hold is a row in a ledger; nothing enforces it, and
nothing observes the breach either -- which is the part that matters, because a
timing run that shared its card produces a plausible number with no sign on it.

**Why KFD and not `rocm-smi`.** Two traps, both hit while writing this:

* `rocm-smi --showpids` has a `GPU(s)` column that is a **count, not an index**.
  Reading it as an index says every process is on GPU 0 or GPU 1 and means
  nothing.
* rocm-smi's device order is not torch's. `scripts/gpu_map.py` reports
  `torch -> rocm-smi {0: 3}` on this node, so "GPU 0" names two different cards
  depending on who is speaking.

`/sys/class/kfd/kfd/proc/<pid>/queues/<q>/gpuid` sidesteps both: the KFD
`gpu_id` is the device's own identifier and is the same number no matter which
tool, container or index space is asking. The authoritative card is identified
by whatever `--owner-pid` is running on, so this needs no index mapping at all
and cannot be fooled by a container renumbering its devices.

Read-only. Samples, records, and says so; it does not kill anything, because
killing somebody else's job on a suspicion is worse than recording the overlap
and letting a human decide which measurement to discard.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import write_artifact  # noqa: E402

KFD_PROC = Path("/sys/class/kfd/kfd/proc")


def gpu_ids_by_pid() -> dict[int, set[str]]:
    """`{pid: {kfd gpu_id, ...}}` for every process with a GPU queue open."""
    out: dict[int, set[str]] = {}
    if not KFD_PROC.is_dir():
        return out
    for proc in KFD_PROC.iterdir():
        try:
            pid = int(proc.name)
        except ValueError:
            continue
        ids = set()
        for q in (proc / "queues").glob("*/gpuid"):
            try:
                ids.add(q.read_text().strip())
            except OSError:
                # The process exited between listing and reading. Not an
                # overlap and not an error.
                continue
        if ids:
            out[pid] = ids
    return out


def cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return "(gone, or another user's)"
    return " ".join(raw.decode("utf-8", "replace").split("\0")).strip() or "(none)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner-pid", type=int, required=True,
                    help="the process whose card is authoritative. Its gpu_id "
                         "IS the definition -- no index mapping involved.")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "11" / "gpu0-guard.json")
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop after this long; 0 means run until the owner "
                         "process exits")
    a = ap.parse_args()

    owner = gpu_ids_by_pid().get(a.owner_pid)
    if not owner:
        print(f"pid {a.owner_pid} has no GPU queue open -- nothing to guard. "
              f"Start this while the authoritative run is already on the card.",
              file=sys.stderr)
        return 2
    if len(owner) > 1:
        print(f"pid {a.owner_pid} spans {sorted(owner)}; guarding all of them",
              file=sys.stderr)

    started = time.time()
    samples = 0
    overlaps: list[dict] = []
    seen: set[tuple[int, str]] = set()
    print(f"guarding gpu_id {sorted(owner)} (owner pid {a.owner_pid})", flush=True)

    while True:
        by_pid = gpu_ids_by_pid()
        if a.owner_pid not in by_pid:
            reason = "owner process left the card"
            break
        for pid, ids in by_pid.items():
            if pid == a.owner_pid:
                continue
            shared = ids & owner
            for gid in sorted(shared):
                key = (pid, gid)
                if key in seen:
                    continue
                seen.add(key)
                row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "pid": pid, "gpu_id": gid, "cmdline": cmdline(pid)[:400]}
                overlaps.append(row)
                print(f"!! OVERLAP on gpu_id {gid}: pid {pid} -- {row['cmdline'][:160]}",
                      file=sys.stderr, flush=True)
        samples += 1
        if a.max_seconds and time.time() - started > a.max_seconds:
            reason = "max-seconds reached"
            break
        time.sleep(a.interval)

    doc = {
        "question": ("Did anything else run on the authoritative card while an "
                     "authoritative timing was in progress? (TODO.md D29)"),
        "owner_pid": a.owner_pid,
        "authoritative_gpu_ids": sorted(owner),
        "method": "/sys/class/kfd/kfd/proc/<pid>/queues/<q>/gpuid, sampled",
        "interval_s": a.interval,
        "samples": samples,
        "watched_seconds": round(time.time() - started, 1),
        "stopped_because": reason,
        "overlaps": overlaps,
        "clean": not overlaps,
        "caveat": ("Sampling, not tracing: a process that opened and closed a "
                   "queue entirely between two samples is not recorded. "
                   f"At {a.interval}s that is a real gap and the figure is "
                   "'no overlap observed', not 'no overlap occurred'."),
    }
    write_artifact(a.out, "11-authoritative-gpu-guard", doc)
    print(f"\n{samples} samples over {doc['watched_seconds']:.0f}s: "
          f"{'CLEAN' if doc['clean'] else str(len(overlaps)) + ' OVERLAP(S)'}"
          f" -- {a.out}")
    return 0 if doc["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
