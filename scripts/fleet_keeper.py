#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Keep every GPU in the fleet working, by relaunching what fell off it.

    python scripts/fleet_keeper.py --plan fleet-plan.json
    python scripts/fleet_keeper.py --plan fleet-plan.json --once --dry-run

``scripts/fleet_monitor.py`` reports idle cards. Reporting is not enough: on
this project the fleet went 71% busy to 8% busy in forty minutes because every
job was finite, each one finished cleanly, and nothing was queued behind it.
The monitor said so the whole time. A human read it twice and a card sat idle
anyway. This closes that loop.

**Why relaunching is safe here, and would not be everywhere.** Every long job
in this repo is (a) *resumable* -- it skips problems whose artifact already
exists -- and (b) *card-pinned*, ``plan[i::N]`` over a sorted plan, so shard i
always covers the same problems on the same GPU. Together those make
relaunching shard i on card i **idempotent**: if it finished, the relaunch
re-checks and exits in seconds; if it died halfway, the relaunch resumes it.
So the keeper does not need to know whether a job succeeded, failed, or was
killed -- only whether the card is working right now.

**Idle is judged by power, not by process.** A wedged process holds the card at
idle power and still appears in ``ps``; a job between problems briefly shows no
driver. Power, sustained across ``--confirm`` consecutive polls, is what
distinguishes "finished" from "between units". A single poll would relaunch on
top of a live job, which for a timing run is the GPU-sharing violation this
project already paid for once.

**The plan is data, not code**, so node names and card assignments stay out of
the repo (they describe a site, not the benchmark). Format::

    {"nodes": {"<hostname or 'localhost'>": {
        "root": "/var/tmp/solbench/m2",
        "gpus": {"0": {"cmd": "...", "log": "..."}, ...}}}}

``cmd`` runs under ``bash -lc`` in ``root``. Use ``{gpu}`` in either field.
A card with no entry is left alone -- that is how you reserve one.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

IDLE_W = 300.0

PROBE = r'''
set -u
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showpower --json 2>/dev/null | python3 -c '
import json,sys,re
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
for k,v in sorted(d.items()):
    if not k.startswith("card"): continue
    idx=re.sub(r"\D","",k); p=0.0
    for kk,vv in v.items():
        if "power" in kk.lower() and "cap" not in kk.lower():
            m=re.search(r"[-+]?[0-9]*\.?[0-9]+", str(vv))
            if m: p=float(m.group())
    print("@@W %s %.0f" % (idx,p))
'
fi
ps -eo args= | grep -E "authoritative_tb|score_solutions|run_agents|verify_anchor|sol_bounds|run_references|derive_tolerances|tb_candidates" \
  | grep -v "docker exec" | grep -v grep | sed 's/^/@@J /'
echo "@@END"
'''


def sh(node: str, script: str, timeout: int = 60) -> str:
    if node in ("localhost", "local"):
        p = subprocess.run(["bash", "-c", script], capture_output=True,
                           text=True, timeout=timeout)
    else:
        p = subprocess.run(["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes",
                            node, "bash -s"], input=script,
                           capture_output=True, text=True, timeout=timeout)
    return p.stdout


def gpus_in_use(jobs: list[str]) -> set[str]:
    """Which GPU indices are claimed by a running job, from its own arguments.

    Read from ``--gpu N`` / ``--gpus a,b`` / ``HIP_VISIBLE_DEVICES``, because a
    job that is mid-compile draws idle power and must still count as owning its
    card. Power says whether the card is working; this says whether anyone has
    claimed it. A relaunch needs BOTH to be clear.
    """
    used: set[str] = set()
    for j in jobs:
        for m in re.finditer(r"--gpus?[= ]([0-9,]+)", j):
            used.update(x for x in m.group(1).split(",") if x)
        for m in re.finditer(r"HIP_VISIBLE_DEVICES[= ]([0-9,]+)", j):
            used.update(x for x in m.group(1).split(",") if x)
    return used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--interval", type=float, default=180.0)
    ap.add_argument("--confirm", type=int, default=2,
                    help="consecutive idle polls before relaunching; 1 will "
                         "relaunch on top of a job that is merely between units")
    ap.add_argument("--idle-watts", type=float, default=IDLE_W)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    plan = json.loads(a.plan.read_text())
    idle_streak: dict[tuple[str, str], int] = defaultdict(int)
    launched: dict[tuple[str, str], int] = defaultdict(int)

    while True:
        stamp = time.strftime("%H:%M:%S")
        for node, spec in (plan.get("nodes") or {}).items():
            root = spec.get("root", "/var/tmp/solbench/m2")
            try:
                out = sh(node, PROBE)
            except Exception as e:  # noqa: BLE001
                print(f"[{stamp}] {node}: probe failed: {e}", flush=True)
                continue
            if "@@END" not in out:
                print(f"[{stamp}] {node}: probe returned nothing usable", flush=True)
                continue

            watts = {}
            jobs = []
            for line in out.splitlines():
                if line.startswith("@@W "):
                    _, i, w = line.split()
                    watts[i] = float(w)
                elif line.startswith("@@J "):
                    jobs.append(line[4:])
            claimed = gpus_in_use(jobs)

            for gpu, entry in (spec.get("gpus") or {}).items():
                key = (node, gpu)
                busy = watts.get(gpu, 0.0) >= a.idle_watts or gpu in claimed
                if busy:
                    idle_streak[key] = 0
                    continue
                idle_streak[key] += 1
                if idle_streak[key] < a.confirm:
                    continue
                cmd = entry["cmd"].replace("{gpu}", gpu)
                log = entry.get("log", f"/var/tmp/solbench/m2-logs/keeper-{gpu}.log")
                log = log.replace("{gpu}", gpu)
                idle_streak[key] = 0
                launched[key] += 1
                n = launched[key]
                print(f"[{stamp}] {node} gpu{gpu}: idle {a.confirm} polls "
                      f"({watts.get(gpu, 0):.0f} W), relaunch #{n}", flush=True)
                if a.dry_run:
                    print(f"           {cmd}", flush=True)
                    continue
                # `cmd` carries leading VAR=value assignments, and nohup cannot
                # take those -- it execs the assignment as a program and dies
                # with "failed to run command 'SOLEXBENCH_CLOCK_BASIS=unlocked'".
                # The keeper counted that as a launch, so it relaunched five
                # jobs on g05 and two survived. Run it through a shell, which is
                # what understands an assignment prefix.
                # The cd, the mkdir and the redirect all live INSIDE the shell,
                # so a log path relative to the tree resolves against the tree
                # rather than against the ssh login directory.
                # No `exec` and no bare `nohup`: neither accepts the leading
                # VAR=value assignments these commands carry. `nohup` execs the
                # assignment as a program ("failed to run command
                # 'SOLEXBENCH_CLOCK_BASIS=unlocked'") and `exec` says
                # "not found". Only a plain shell line understands the prefix.
                inner = (f"cd {shlex.quote(root)} && mkdir -p "
                         f"$(dirname {shlex.quote(log)}) && "
                         f"{cmd} >> {shlex.quote(log)} 2>&1")
                full = (f"setsid nohup bash -lc {shlex.quote(inner)} "
                        f">/dev/null 2>&1 < /dev/null & disown; true")
                try:
                    sh(node, full, timeout=90)
                except Exception as e:  # noqa: BLE001
                    print(f"           launch failed: {e}", flush=True)
                    continue
                # Verify, do not assume. The first version of this keeper
                # reported ten successful relaunches while nohup was killing
                # every one of them on a leading VAR=value assignment; the
                # count went up and the fleet stayed idle. A launcher that
                # cannot see its own failures is the same defect as a gate
                # passing over an empty list.
                # 20s, not 4s: the wrapper shell is alive immediately and the
                # job inside it dies a moment later, so a short check confirms
                # the launcher rather than the job. Checking too early is how
                # this reported eleven live relaunches while every one of them
                # had already failed.
                time.sleep(20)
                alive = sh(node, "ps -eo args= | grep -v grep | grep -c "
                                 f"-- {shlex.quote('--gpu ' + gpu)} || true")
                if not alive.strip().strip("0"):
                    tailed = sh(node, f"tail -3 {shlex.quote(log)} 2>/dev/null || true",
                                timeout=30).strip()
                    print(f"           DID NOT START. log tail: {tailed[:300]}",
                          flush=True)

        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
