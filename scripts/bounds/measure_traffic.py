#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure the DRAM bytes a reference actually moves, and falsify a traffic bound.

    python scripts/bounds/measure_traffic.py --gpus 1,2,3,4,5,6,7 \
        --out artifacts/11/measured-traffic.json

**Why this needs a GPU when SOL bounds do not.** D42 established, by hand, that
the declared-traffic tier prices inputs the kernel never reads. Deriving the
replacement by hand and then checking it by hand repeats exactly the mistake
`CLAUDE.md` s6 warns about: a self-consistent bound and anchor cannot detect a
shared error. This is the independent number.

**It is a falsification test, and that is deliberate.** What it measures is the
traffic of the problem's own PyTorch *reference*, which is an upper bound on
what a competent kernel must move -- the reference may itself be wasteful, and
several here demonstrably are. So:

    measured_reference_bytes  <  tier_bytes   =>  the tier over-counts, proven
    measured_reference_bytes >= tier_bytes    =>  says nothing either way

Only the first direction is a result. A problem that does not falsify is not
thereby cleared, and this script says so per problem rather than printing a
tick. That asymmetry is the whole point: it cannot manufacture a bound, only
rule one out.

**GPU discipline.** Defaults to cards 1-7 and REFUSES card 0, which is for
authoritative timing only (`CLAUDE.md` s4). Nothing here is a timing
measurement -- counters change kernel duration -- and no number it produces may
be used as one.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data" / "SOL-ExecBench" / "benchmark"
SCRATCH = Path(os.environ.get("SOLEXBENCH_SCRATCH", "/var/tmp/solbench"))

#: bytes/cycle at F_LOCK, restated rather than imported -- see the note in
#: diagnose_bad_bounds.py about a check that shares the value it checks.
DRAM_BYTE_PER_CYCLE = 6153.8

_lock = threading.Lock()


def say(msg: str) -> None:
    with _lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


DRIVER = r'''
import json, os, sys, importlib.util
import torch

problem = sys.argv[1]
uuid    = sys.argv[2]

spec = importlib.util.spec_from_file_location("ref", os.path.join(problem, "reference.py"))
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)

# The axes in workload.jsonl are the VAR axes only; const and expr axes live in
# definition.json and get_inputs expects all of them. Resolved through the
# harness's own Definition rather than re-merged here -- a second
# implementation of that merge would drift from the one that produced every
# measured number in the repo, silently and in the shapes.
from sol_execbench.core import Definition, Workload
definition = Definition(**json.load(open(os.path.join(problem, "definition.json"))))
wl = None
with open(os.path.join(problem, "workload.jsonl")) as fh:
    for line in fh:
        d = json.loads(line)
        if d["uuid"] == uuid:
            wl = Workload(**d); break
if wl is None:
    raise SystemExit(f"no workload {uuid}")

axes = {**definition.get_resolved_axes_values(wl.axes), **wl.get_scalar_inputs()}
dev = torch.device("cuda:0")
kwargs = dict(ref.get_inputs(axes, dev))
torch.cuda.synchronize()

# Warm up OUTSIDE the counted region; the counters are collected over the whole
# process, so a compile or an allocator growth inside it would be counted as
# traffic the workload moved.
for _ in range(3):
    ref.run(**kwargs)
torch.cuda.synchronize()

REPS = int(os.environ.get("TRAFFIC_REPS", "10"))
print("COUNTED_REGION_BEGIN", flush=True)
for _ in range(REPS):
    ref.run(**kwargs)
torch.cuda.synchronize()
print("COUNTED_REGION_END", flush=True)
print(json.dumps({"reps": REPS, "axes": axes}))
'''


def measure(problem_dir: Path, uuid: str, gpu: int, reps: int) -> dict:
    """rocprofv3 FETCH_SIZE + WRITE_SIZE over `reps` calls of the reference."""
    work = SCRATCH / "traffic" / f"{problem_dir.name}-{uuid[:8]}"
    work.mkdir(parents=True, exist_ok=True)
    drv = work / "drv.py"
    drv.write_text(DRIVER)

    cmd = [str(ROOT / "env" / "solb"), "rocprofv3",
           "--pmc", "FETCH_SIZE", "WRITE_SIZE",
           "-d", str(work), "--output-format", "csv", "-o", "pmc",
           "--", "python", str(drv),
           f"/work/data/SOL-ExecBench/benchmark/{problem_dir.parent.name}/{problem_dir.name}",
           uuid]
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home()),
           "HIP_VISIBLE_DEVICES": str(gpu), "TRAFFIC_REPS": str(reps)}
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=900)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timed out"}
    if proc.returncode != 0:
        return {"ok": False, "error": f"rc={proc.returncode}",
                "stderr_tail": proc.stderr[-1500:]}

    rows = []
    for csvf in work.rglob("*pmc*.csv"):
        with csvf.open() as fh:
            rows.extend(list(csv.DictReader(fh)))
    if not rows:
        return {"ok": False, "error": "rocprofv3 produced no counter rows",
                "stderr_tail": proc.stderr[-1500:]}

    fetch = write = 0.0
    kernels = 0
    for r in rows:
        name = (r.get("Counter_Name") or r.get("Counter") or "").strip()
        try:
            val = float(r.get("Counter_Value") or r.get("Value") or 0)
        except ValueError:
            continue
        if name == "FETCH_SIZE":
            fetch += val; kernels += 1
        elif name == "WRITE_SIZE":
            write += val
    if kernels == 0:
        return {"ok": False, "error": "no FETCH_SIZE rows in counter output"}

    # rocprof reports these in KB.
    return {"ok": True,
            "fetch_bytes_total": fetch * 1024,
            "write_bytes_total": write * 1024,
            "kernel_dispatches": kernels,
            "reps_including_warmup": reps + 3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,2,3,4,5,6,7")
    ap.add_argument("--manifest", default="artifacts/09/manifest-v1.2.json")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--problems", nargs="*", default=None,
                    help="default: every problem whose bound rests on the "
                         "declared-traffic tier on at least one workload")
    ap.add_argument("--out", default="artifacts/11/measured-traffic.json")
    a = ap.parse_args()

    gpus = [int(g) for g in a.gpus.split(",") if g.strip()]
    if 0 in gpus:
        raise SystemExit("card 0 is for authoritative timing only (CLAUDE.md s4). "
                         "Counters perturb kernel duration; this must not share "
                         "the card a published number is measured on.")

    man = json.loads((ROOT / a.manifest).read_text())["problems"]

    # One workload per problem: the one where the traffic tier wins by the
    # widest margin, which is where a falsification is most likely to be
    # unambiguous. Measuring all sixteen would cost 16x for a result that is
    # per-problem either way.
    targets = []
    for key, prob in sorted(man.items()):
        if a.problems and key not in a.problems:
            continue
        best = None
        for uuid, w in prob["workloads"].items():
            s, t = w.get("t_sol_cycles_solar"), w.get("t_sol_cycles_traffic")
            if not (s and t and t > s and w.get("scoreable")):
                continue
            if best is None or t / s > best[2]:
                best = (uuid, w, t / s)
        if best:
            targets.append((key, *best))
    targets.sort(key=lambda x: -x[3])

    say(f"{len(targets)} problems bounded by the traffic tier, "
        f"{len(gpus)} cards (0 excluded)")

    q: queue.Queue = queue.Queue()
    for t in targets:
        q.put(t)
    out: list[dict] = []

    def worker(gpu: int) -> None:
        while True:
            try:
                key, uuid, w, ratio = q.get_nowait()
            except queue.Empty:
                return
            cat, name = key.split("__", 1)
            t0 = time.time()
            m = measure(BENCH / cat / name, uuid, gpu, a.reps)
            rec = {"problem": key, "workload_uuid": uuid, "gpu": gpu,
                   "tier_over_solar": round(ratio, 3),
                   "t_sol_cycles_traffic": w.get("t_sol_cycles_traffic"),
                   "t_sol_cycles_solar": w.get("t_sol_cycles_solar"),
                   **m}
            if m.get("ok"):
                per_rep = ((m["fetch_bytes_total"] + m["write_bytes_total"])
                           / rec["reps_including_warmup"])
                tier_bytes = w["t_sol_cycles_traffic"] * DRAM_BYTE_PER_CYCLE
                rec.update({
                    "measured_bytes_per_call": per_rep,
                    "tier_bytes": tier_bytes,
                    "tier_over_measured": round(tier_bytes / per_rep, 3) if per_rep else None,
                    # The ONLY conclusion this script may draw.
                    "falsifies_the_tier": bool(per_rep and tier_bytes > per_rep),
                })
            with _lock:
                out.append(rec)
            say(f"gpu{gpu} {key[:46]:46} {time.time()-t0:5.0f}s  "
                + (f"tier/measured {rec.get('tier_over_measured')}"
                   if m.get("ok") else f"FAILED {m.get('error')}"))

    threads = [threading.Thread(target=worker, args=(g,), daemon=True)
               for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = [r for r in out if r.get("ok")]
    falsified = [r for r in ok if r.get("falsifies_the_tier")]
    payload = {
        "question": "D42 says the declared-traffic tier prices bytes the kernel "
                    "never moves. Measured independently, does the problem's "
                    "own reference move fewer bytes than the tier prices?",
        "method": "rocprofv3 FETCH_SIZE + WRITE_SIZE over N calls of the "
                  "reference, warmup outside the counted region, one workload "
                  "per problem (the one where the tier beats SOLAR widest), "
                  "cards 1-7, card 0 refused.",
        "one_sided": "measured < tier PROVES the tier over-counts. measured >= "
                     "tier proves NOTHING -- the reference is an upper bound on "
                     "required traffic, not a lower one. A problem that does "
                     "not falsify is not cleared.",
        "not_a_timing_measurement": "counters perturb kernel duration. No "
                                    "number here may be used as a T_k or a T_b.",
        "problems_measured": len(ok),
        "problems_failed": len(out) - len(ok),
        "tier_falsified_on": len(falsified),
        "results": sorted(out, key=lambda r: -(r.get("tier_over_measured") or 0)),
    }

    sys.path.insert(0, str(ROOT / "scripts"))
    from provenance import write_artifact                     # noqa: E402
    dest = ROOT / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_artifact(dest, "11-measured-traffic", payload)
    say(f"wrote {dest}")
    say(f"{len(ok)} measured, {len(falsified)} falsify the tier, "
        f"{len(out)-len(ok)} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
