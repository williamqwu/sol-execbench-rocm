#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Watch the authoritative GPU and record anything that lands on it.

    python scripts/guard_authoritative_gpu.py --gpu-id 36538 \
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
tool, container or index space is asking -- so this needs no index mapping at
all and cannot be fooled by a container renumbering its devices. Name the card
with `--gpu-id`, or resolve it once from a process with `--owner-pid`.

**The invariant is "one job at a time on this card", not "this pid is here".**
The first version of this guard tracked a pid, and `agent_score.py` shells each
problem into its own container -- so the pid it was watching left the card after
one problem out of forty, the guard concluded the run was over, and it reported
CLEAN on 45 seconds of a two-hour window. Watching the card survives the worker
rotating; watching a worker does not.

Read-only. Samples, records, and says so; it does not kill anything, because
killing somebody else's job on a suspicion is worse than recording the overlap
and letting a human decide which measurement to discard.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
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


def ppid(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def container(pid: int) -> str:
    """Which cgroup the process is in -- enough to tell a fleet job from ours."""
    try:
        text = Path(f"/proc/{pid}/cgroup").read_text()
    except OSError:
        return "(unreadable)"
    for line in text.splitlines():
        tail = line.rsplit(":", 1)[-1]
        if "docker" in tail or "containerd" in tail or "kubepods" in tail:
            return tail.strip()[-80:]
    return "host"


def cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return "(gone, or another user's)"
    return " ".join(raw.decode("utf-8", "replace").split("\0")).strip() or "(none)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner-pid", type=int, default=None,
                    help="resolve the authoritative card from whatever this "
                         "process is running on, once, at startup. Its gpu_id "
                         "IS the definition -- no index mapping involved.")
    ap.add_argument("--gpu-id", default=None,
                    help="the KFD gpu_id to guard, if it is already known. "
                         "Prefer this for a run whose worker process ROTATES: "
                         "`agent_score.py` shells each problem into its own "
                         "container, so any single pid leaves the card between "
                         "problems and an owner-pid guard stops after the "
                         "first one. That is how the first version of this "
                         "guard watched 45 seconds of a two-hour re-time and "
                         "reported CLEAN.")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "11" / "gpu0-guard.json")
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop after this long; 0 means run until stopped")
    ap.add_argument("--until-idle", action="store_true",
                    help="stop once nothing at all is on the card. Off by "
                         "default: a run that shells one container per problem "
                         "leaves the card briefly between them, and stopping "
                         "there ends the guard in the first gap.")
    a = ap.parse_args()

    if a.gpu_id:
        owner = {a.gpu_id}
    elif a.owner_pid:
        owner = gpu_ids_by_pid().get(a.owner_pid) or set()
        if not owner:
            print(f"pid {a.owner_pid} has no GPU queue open -- nothing to "
                  f"resolve. Start this while the run is already on the card, "
                  f"or pass --gpu-id.", file=sys.stderr)
            return 2
        if len(owner) > 1:
            print(f"pid {a.owner_pid} spans {sorted(owner)}; guarding all",
                  file=sys.stderr)
    else:
        print("need --gpu-id or --owner-pid", file=sys.stderr)
        return 2

    started = time.time()
    samples = 0
    overlaps: list[dict] = []
    seen: set[tuple[int, str]] = set()
    print(f"guarding gpu_id {sorted(owner)} (owner pid {a.owner_pid})", flush=True)

    reason = "interrupted"
    occupants: dict[str, set[int]] = {}

    # Write the artifact on the way out, including when something kills us.
    # The first run of this guard was ended with SIGTERM and produced no file
    # at all -- eleven hours of sampling, five real detections, and the only
    # record was whatever had been printed to a log. A watcher whose evidence
    # depends on it exiting politely is a watcher you will lose exactly when
    # you needed it.
    stopping: list[str] = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, lambda s, _f: stopping.append(f"signal {s}"))
        except ValueError:
            pass
    while True:
        by_pid = gpu_ids_by_pid()
        # The invariant is not "this pid is here". It is "one job at a time on
        # this card" -- which is what makes an authoritative timing
        # authoritative, and which survives the timing run rotating pids.
        for gid in owner:
            here = sorted(p for p, ids in by_pid.items() if gid in ids)
            occupants.setdefault(gid, set()).update(here)
            if len(here) > 1:
                key = (tuple(here), gid)
                if key in seen:
                    continue
                seen.add(key)
                # Captured AT DETECTION, not at exit. The first version kept
                # cmdlines only for the final artifact and printed bare pids,
                # so by the time an overlap was noticed the processes were gone
                # and there was no way to tell a fleet job from this run's own
                # container teardown. An alert that cannot say what it saw
                # leaves you guessing, which is worse than not alerting.
                who = {p: {"cmdline": cmdline(p)[:300], "ppid": ppid(p),
                           "cgroup": container(p)} for p in here}
                row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "gpu_id": gid, "pids": here, "processes": who}
                overlaps.append(row)
                print(f"!! {row['utc']}  {len(here)} processes on gpu_id {gid}:",
                      file=sys.stderr)
                for pp, info in who.items():
                    print(f"     pid {pp} ppid={info['ppid']} "
                          f"cgroup={info['cgroup']}\n       {info['cmdline'][:200]}",
                          file=sys.stderr)
                sys.stderr.flush()
        samples += 1
        if stopping:
            reason = stopping[0]
            break
        if a.max_seconds and time.time() - started > a.max_seconds:
            reason = "max-seconds reached"
            break
        if a.until_idle and not any(
                any(gid in ids for gid in owner) for ids in by_pid.values()):
            reason = "card went idle"
            break
        try:
            time.sleep(a.interval)
        except InterruptedError:
            pass

    doc = {
        "question": ("Did anything else run on the authoritative card while an "
                     "authoritative timing was in progress? (TODO.md D29)"),
        "owner_pid": a.owner_pid,
        "authoritative_gpu_ids": sorted(owner),
        "method": "/sys/class/kfd/kfd/proc/<pid>/queues/<q>/gpuid, sampled",
        "interval_s": a.interval,
        "samples": samples,
        "distinct_pids_seen_on_card": {g: sorted(v) for g, v in occupants.items()},
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
