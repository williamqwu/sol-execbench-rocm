#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""D20: find what makes one matmul iteration in ~100 run 3x-21x slow.

    env/solb python scripts/probe_timing_stall.py --gpus 1 2 3 4

`artifacts/02/timing-variance-amd.json` established the effect and ruled out
the obvious cause. It is not cold start -- the first call of every size is the
*tightest* sample taken. The outliers are too tightly clustered to be jitter:
21.4 / 21.6 / 22.5x on three different GPUs at mm[2048], and 3.3-3.4x nine
separate times at mm[4096]. That repeatability is the thing to explain.

The previous script only kept `max/min`, which throws away the one fact that
discriminates between the hypotheses: **where in the iteration sequence the
stall lands.**

  * fixed index, stable across seeds -> structural. `time_runnable` builds a
    ShiftingMemoryPoolAllocator over warmup+rep iterations and hands each one a
    differently-shifted pointer; at 2048x2048 fp32 that pool is gigabytes, so a
    pool wrap, a page boundary or a TLB reach limit would show up at a
    reproducible offset.
  * fixed index that MOVES with the seed -> also the allocator, since the seed
    is exactly what randomizes the pointer-shift sequence.
  * random index -> environmental: a power/clock excursion, driver work, or
    queue preemption. Those cannot be pinned from inside the process, so that
    outcome routes to an external clock trace instead of more of this.

Ablations, each isolating one candidate:

  seed      changes the pointer-shift sequence, nothing else
  rep       if stalls-per-run scales with rep, it is a per-iteration hazard;
            if it stays at ~1, it is a per-call event
  pool      a shorter run means a smaller pool; if the effect tracks pool size
            rather than iteration count, it is memory, not time

Reports raw counts and indices. It does not decide -- there is not yet enough
evidence to name a cause, and naming one anyway is how D20 would become a
plausible number nobody measured.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# 2048 and 4096 carry the tight clusters; 64 is the diffuse launch-bound case
# and is kept as a contrast, not as a target.
SIZES = [64, 2048, 4096]
OUTLIER = 2.0          # an iteration this many times the median is "a stall"


def worker(gpu: int, calls: int, reps: list[int], seeds: list[int]) -> dict:
    code = f'''
import json, statistics, torch
from sol_execbench.core.bench.timing import time_runnable

out = []
for size in {SIZES!r}:
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    for rep in {reps!r}:
        for seed in {seeds!r}:
            for call in range({calls}):
                t = time_runnable(lambda a, b: torch.mm(a, b), [a, b], [],
                                  "cuda:0", rep=rep, seed=seed,
                                  return_mode="all")
                med = statistics.median(t)
                idx = [i for i, x in enumerate(t) if x > {OUTLIER} * med]
                out.append({{"size": size, "rep": rep, "seed": seed,
                             "call": call, "n": len(t),
                             "median_ms": med,
                             "max_ms": max(t),
                             "ratio": max(t) / min(t) if min(t) > 0 else None,
                             "stall_idx": idx,
                             "stall_vals": [round(t[i] / med, 2) for i in idx]}})
    del a, b
    torch.cuda.empty_cache()
print("@@@" + json.dumps(out))
'''
    proc = subprocess.run(
        [str(ROOT / "env" / "solb"), "python", "-c", code],
        capture_output=True, text=True, timeout=5400,
        env={**os.environ, "HIP_VISIBLE_DEVICES": str(gpu)})
    if proc.returncode != 0:
        raise SystemExit(f"gpu {gpu} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    line = [l for l in proc.stdout.splitlines() if l.startswith("@@@")]
    if not line:
        raise SystemExit(f"gpu {gpu}: no result\n{proc.stdout[-3000:]}")
    return json.loads(line[-1][3:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--calls", type=int, default=12, help="invocations per cell")
    ap.add_argument("--reps", type=int, nargs="+", default=[100, 25])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 7])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "02" / "timing-stall-probe.json")
    a = ap.parse_args()
    if 0 in a.gpus:
        raise SystemExit("GPU 0 is reserved for authoritative timing; use 1-7.")

    rows = []
    for g in a.gpus:
        print(f"  gpu {g} ...", flush=True)
        for r in worker(g, a.calls, a.reps, a.seeds):
            r["gpu"] = g
            rows.append(r)

    findings = {}
    for size in SIZES:
        s_rows = [r for r in rows if r["size"] == size]
        cells = {}
        for rep in a.reps:
            for seed in a.seeds:
                cell = [r for r in s_rows if r["rep"] == rep and r["seed"] == seed]
                stalls = [(r["gpu"], i, v) for r in cell
                          for i, v in zip(r["stall_idx"], r["stall_vals"])]
                idx_hist = Counter(i for _, i, _ in stalls)
                cells[f"rep{rep}_seed{seed}"] = {
                    "calls": len(cell),
                    "calls_with_a_stall": sum(1 for r in cell if r["stall_idx"]),
                    "total_stalls": len(stalls),
                    "stalls_per_call": round(len(stalls) / len(cell), 3) if cell else None,
                    "median_ms": round(statistics.median(
                        [r["median_ms"] for r in cell]), 6) if cell else None,
                    "worst_ratio": round(max((r["ratio"] or 0) for r in cell), 2) if cell else None,
                    "index_histogram": dict(sorted(idx_hist.items())[:20]),
                    "distinct_indices": len(idx_hist),
                    "index_concentration": (
                        round(max(idx_hist.values()) / sum(idx_hist.values()), 3)
                        if idx_hist else None),
                }
        findings[str(size)] = cells
        print(f"\n  mm[{size}]")
        for name, c in cells.items():
            print(f"    {name:16s} stalls/call {str(c['stalls_per_call']):>6s}  "
                  f"worst {str(c['worst_ratio']):>6s}x  "
                  f"distinct idx {c['distinct_indices']:>3d}  "
                  f"concentration {c['index_concentration']}")

    from provenance import write_artifact
    write_artifact(a.out, "02-timing-stall-probe", {
        "_note": "D20 discrimination probe. Records WHERE in the iteration "
                 "sequence a slow iteration lands, across seeds and rep counts, "
                 "to separate an allocator/memory cause from an environmental "
                 "one. Reports evidence only; it does not assign a cause.",
        "outlier_definition": f"iteration > {OUTLIER}x the call's median",
        "gpus": a.gpus, "calls_per_cell": a.calls,
        "reps": a.reps, "seeds": a.seeds,
        "findings": findings,
        "raw": rows,
    })
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
