#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fleet monitor -- what every GPU on every node is doing, on one screen.

    python scripts/fleet_monitor.py --nodes localhost,host-b,host-c
    python scripts/fleet_monitor.py --nodes ... --watch 30

Written because the failure this project actually keeps having is not a wrong
number, it is an *idle card*: a sweep that died at 3am, a shard that finished
and freed a GPU nobody refilled, a node whose log directory did not exist so
the job never started. None of those raise. All of them look exactly like
"work is in progress" from the shell you happen to be sitting in.

So the monitor answers three questions per node, in this order:

  1. Which cards are BUSY, by power draw and clock -- not by whether a process
     exists. A wedged process holds a GPU at idle power and still shows up in
     ``ps``. Power is the honest signal.
  2. Which solbench jobs are running, with their shard and their age. An
     ``eval_driver.py`` whose parent is gone is an orphan (D34) and is called
     out, because it holds a card and nothing will ever collect its result.
  3. How far the artifacts have got -- file counts under the trees that the
     current phase writes into. Progress, not liveness.

Nothing here is part-specific or host-specific: pass ``--nodes``. Idle-card
detection uses power because the clock alone is misleading (a memory-bound
kernel and an idle card can sit at similar sclk on this part).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Power floor below which a card is doing nothing worth having reserved. On
# MI355X an idle card sits near 240-250 W and any real kernel is several
# hundred. Deliberately not a utilisation percentage: rocm-smi's use% reads
# 100% for a card spinning on a single wave.
IDLE_W = 300.0

# The scripts worth naming. Anything else running under the tree is shown as
# "other" rather than hidden, because an unrecognised job holding GPU 0 is
# precisely the thing you want to see.
KNOWN = {
    "authoritative_tb.py": "T_b auth",
    "score_solutions.py": "score",
    "run_agents.py": "agents",
    "verify_anchor.py": "anchor",
    "sol_bounds.py": "bounds",
    "run_references.py": "refs",
    "derive_tolerances.py": "tol",
    "tb_candidates.py": "T_b cand",
    "eval_driver.py": "  driver",
}

REMOTE = r'''
set -u
ROOT="%ROOT%"
echo "@@HOST $(hostname -s)"
# --- cards: index, sclk MHz, power W, vram used MiB ---
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showpower --showclocks --showmemuse --json 2>/dev/null \
    | python3 -c '
import json,sys,re
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
def num(s):
    m=re.search(r"[-+]?[0-9]*\.?[0-9]+", str(s))
    return float(m.group()) if m else 0.0
for k,v in sorted(d.items()):
    if not k.startswith("card"): continue
    idx=re.sub(r"\D","",k)
    p=sclk=mem=0.0
    for kk,vv in v.items():
        lk=kk.lower()
        if "power" in lk and "cap" not in lk: p=num(vv)
        # "sclk clock level" is an INDEX ("1"); the speed lives in the value
        # of "sclk clock speed" as "(2397Mhz)". Matching "sclk" alone reads
        # the index and reports every card at 1 MHz -- which is what the
        # first version of this monitor did.
        elif "sclk" in lk and "level" not in lk: sclk=num(vv)
        elif "sclk" in lk and "Mhz" in str(vv): sclk=num(vv)
        elif ("vram" in lk or "memory" in lk) and ("used" in lk or "use" in lk):
            mem=num(vv)
    print("@@GPU %s %.0f %.0f %.0f" % (idx, sclk, p, mem))
'
fi
# Every live pid, so orphan detection can ask whether a process group leader
# still exists at all. Asking only against the matched jobs says "orphan" for
# any driver whose parent is not itself a driver, which is the normal case.
echo "@@PIDS $(ps -eo pid= | tr -d ' ' | tr '\n' ',')"
# --- jobs: pid pgid etime cmd, one line each, non-docker-exec only ---
ps -eo pid=,pgid=,etimes=,args= | grep -E "authoritative_tb|score_solutions|run_agents|verify_anchor|sol_bounds|run_references|derive_tolerances|tb_candidates|eval_driver" \
  | grep -v "docker exec" | grep -v "grep -E" | while read -r pid pgid et rest; do
    echo "@@JOB $pid $pgid $et $rest"
  done
# --- artifact progress ---
for d in 06-MI355X/authoritative 06-MI355X/authoritative-40 06-MI355X/authoritative-g05 \
         06-MI355X/authoritative-merged 06-MI355X/candidates 02-MI355X/references-amd \
         05-MI355X/workloads 10/scores; do
  if [ -d "$ROOT/artifacts/$d" ]; then
    n=$(find "$ROOT/artifacts/$d" -name '*.json' 2>/dev/null | wc -l)
    echo "@@ART $d $n"
  fi
done
for r in "$ROOT"/artifacts/10/runs/*/; do
  [ -d "$r" ] || continue
  n=$(find "$r" -name session.json 2>/dev/null | wc -l)
  echo "@@RUN $(basename "$r") $n"
done
echo "@@END"
'''


def probe(node: str, root: str, timeout: int) -> dict:
    """One node. Never raises -- an unreachable node is a finding, not a crash."""
    script = REMOTE.replace("%ROOT%", root)
    if node in ("localhost", "local", ""):
        cmd = ["bash", "-c", script]
    else:
        cmd = ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes",
               node, "bash -s"]
    try:
        p = subprocess.run(cmd, input=script if node not in ("localhost", "local", "") else None,
                           capture_output=True, text=True, timeout=timeout)
        out = p.stdout
    except subprocess.TimeoutExpired:
        return {"node": node, "error": f"probe timed out after {timeout}s"}
    except Exception as e:  # noqa: BLE001
        return {"node": node, "error": str(e)}
    if "@@END" not in out:
        return {"node": node, "error": (p.stderr or out or "no output").strip()[:200]}

    r: dict = {"node": node, "host": node, "gpus": [], "jobs": [], "art": {},
               "runs": {}, "pids": set()}
    for line in out.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "@@HOST" and len(f) > 1:
            r["host"] = f[1]
        elif f[0] == "@@PIDS" and len(f) > 1:
            r["pids"] = {int(x) for x in f[1].split(",") if x.strip().isdigit()}
        elif f[0] == "@@GPU" and len(f) >= 5:
            r["gpus"].append({"i": int(f[1]), "sclk": float(f[2]),
                              "w": float(f[3]), "mem": float(f[4])})
        elif f[0] == "@@JOB" and len(f) >= 5:
            r["jobs"].append({"pid": int(f[1]), "pgid": int(f[2]),
                              "etimes": int(f[3]), "cmd": " ".join(f[4:])})
        elif f[0] == "@@ART" and len(f) >= 3:
            r["art"][f[1]] = int(f[2])
        elif f[0] == "@@RUN" and len(f) >= 3:
            r["runs"][f[1]] = int(f[2])
    return r


def label(cmd: str) -> tuple[str, str]:
    """(short name, distinguishing detail) for a command line."""
    for key, name in KNOWN.items():
        if key in cmd:
            bits = []
            toks = shlex.split(cmd) if "'" not in cmd else cmd.split()
            for flag in ("--shard", "--gpu", "--run-id", "--gpus"):
                if flag in toks:
                    bits.append(f"{flag.lstrip('-')}={toks[toks.index(flag) + 1]}")
            return name, " ".join(bits)
    return "other", cmd[:60]


def hms(s: int) -> str:
    h, rem = divmod(s, 3600)
    return f"{h}:{rem // 60:02d}" if h else f"{rem // 60}m"


def render(rows: list[dict]) -> str:
    out: list[str] = []
    ts = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    busy_t = idle_t = 0
    body: list[str] = []

    for r in rows:
        if "error" in r:
            body.append(f"\n=== {r['node']}  UNREACHABLE: {r['error']}")
            continue
        gpus = r["gpus"]
        busy = [g for g in gpus if g["w"] >= IDLE_W]
        idle = [g for g in gpus if g["w"] < IDLE_W]
        busy_t += len(busy)
        idle_t += len(idle)
        body.append(f"\n=== {r['host']}  {len(busy)}/{len(gpus)} busy"
                    + (f"   IDLE: {','.join(str(g['i']) for g in idle)}" if idle else ""))
        if gpus:
            body.append("  gpu " + " ".join(f"{g['i']:>6}" for g in gpus))
            body.append("  MHz " + " ".join(f"{g['sclk']:>6.0f}" for g in gpus))
            body.append("  W   " + " ".join(f"{g['w']:>6.0f}" for g in gpus))
            body.append("  GiB " + " ".join(f"{g['mem'] / 1024:>6.1f}" for g in gpus))

        live = r.get("pids") or set()
        for j in sorted(r["jobs"], key=lambda x: -x["etimes"]):
            name, detail = label(j["cmd"])
            # An eval_driver whose process-group leader is gone is D34: it holds
            # a card and nobody will ever collect what it produces. Checked
            # against EVERY live pid -- against the matched jobs only, every
            # driver whose parent is not itself a driver reads as an orphan,
            # which is the normal case and was this monitor's first false alarm.
            orphan = ("eval_driver" in j["cmd"] and live
                      and j["pgid"] not in live and j["pgid"] != j["pid"])
            flag = "  <-- ORPHAN (D34)" if orphan else ""
            body.append(f"  {hms(j['etimes']):>6}  {name:<9} {detail}{flag}")
        if not r["jobs"]:
            body.append("  (no solbench jobs)")

        prog = [f"{k.split('/')[-1]}={v}" for k, v in sorted(r["art"].items())]
        if prog:
            body.append("  art: " + "  ".join(prog))
        runs = [f"{k}={v}" for k, v in sorted(r["runs"].items())]
        if runs:
            body.append("  runs: " + "  ".join(runs))

    tot = busy_t + idle_t
    pct = 100.0 * busy_t / tot if tot else 0.0
    out.append(f"FLEET {ts}   {busy_t}/{tot} cards busy ({pct:.0f}%)"
               f"   idle threshold {IDLE_W:.0f} W")
    out.extend(body)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", required=True,
                    help="comma-separated; use 'localhost' for this one")
    ap.add_argument("--root", default="/var/tmp/solbench/m2",
                    help="working tree on each node")
    ap.add_argument("--watch", type=float, metavar="SECONDS",
                    help="re-probe forever at this interval")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--json", action="store_true", help="emit raw records")
    ap.add_argument("--out", type=Path, help="also append each render here")
    args = ap.parse_args()

    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    while True:
        with cf.ThreadPoolExecutor(max_workers=len(nodes)) as ex:
            rows = list(ex.map(lambda n: probe(n, args.root, args.timeout), nodes))
        text = json.dumps(rows, indent=2) if args.json else render(rows)
        print(text, flush=True)
        if args.out:
            with args.out.open("a") as fh:
                fh.write(text + "\n")
        if not args.watch:
            return 0
        print("", flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
