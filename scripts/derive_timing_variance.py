#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Re-derive the timing-variance thresholds in `test_timing.py` on AMD.

    env/solb python scripts/derive_timing_variance.py --gpus 1 2 3 4 --reps 30

`TestTimeRunnable::test_matmul_timing_variance` asserts that the trimmed
min/max spread of `time_runnable` stays under a per-size constant. Those
constants -- 1.25, 1.30, 1.15, 1.15 -- are NVIDIA numbers: the test's own
docstring cites "measured ranges on RTX 4090 and B200". Prime directive 2 says
they may not be carried into an AMD artifact, and in fact they do not hold
here: mm[64x64] exceeded 1.25x on 5 of 5 runs.

What this script does NOT do is change how the quantity is measured. The
statistic stays exactly what the test computes -- `max(times)/min(times)` over
one `time_runnable(..., return_mode="all")` call, with the same default warmup
and rep counts -- because changing the statistic to something better behaved
would be a methodology change made to get past an obstacle (directive 7). Only
the constant is re-derived.

`max/min` over a trimmed sample is an extreme-order statistic, so its own
distribution is heavy-tailed: a single dispatch hiccup moves it. That is why
the threshold comes from the observed maximum over many independent
invocations rather than from a mean or a median, and why it carries explicit
margin on top. The margin is stated in the artifact, not folded silently into
a rounded number.

Measured across several GPUs on purpose. The eight cards on this node do not
hold the same clock at the same determinism setting (they span 1242-1307 MHz),
so a threshold derived on one card can fail on another. GPU 0 is left out: it
is reserved for authoritative timing, and tests run wherever a developer sends
them, which is the non-authoritative cards.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Sizes and the NVIDIA constants they are replacing, from the parametrize list
# in TestTimeRunnable::test_matmul_timing_variance.
CASES = [(64, 1.25), (512, 1.30), (2048, 1.15), (4096, 1.15)]

# Headroom over the worst spread actually observed. The test is a regression
# guard, not a specification: it must not flake on healthy hardware, and it
# must still catch a kernel or a driver that has become genuinely unstable.
MARGIN = 1.15


def worker(gpu: int, reps: int) -> dict:
    """Measure in a fresh process per GPU, so no CUDA context is reused."""
    code = f'''
import json, torch
from sol_execbench.core.bench.timing import time_runnable
out = {{}}
for size in {[c[0] for c in CASES]!r}:
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    ratios, cvs = [], []
    for _ in range({reps}):
        t = time_runnable(lambda a, b: torch.mm(a, b), [a, b], [], "cuda:0",
                          return_mode="all")
        ratios.append(max(t) / min(t) if min(t) > 0 else float("inf"))
        m = sum(t) / len(t)
        v = sum((x - m) ** 2 for x in t) / len(t)
        cvs.append((v ** 0.5) / m if m > 0 else float("inf"))
    out[str(size)] = {{"ratios": ratios, "cvs": cvs}}
    del a, b
    torch.cuda.empty_cache()
print("@@@" + json.dumps(out))
'''
    env_cmd = [str(ROOT / "env" / "solb"), "python", "-c", code]
    proc = subprocess.run(env_cmd, capture_output=True, text=True, timeout=3600,
                          env={**__import__("os").environ,
                               "HIP_VISIBLE_DEVICES": str(gpu)})
    if proc.returncode != 0:
        raise SystemExit(f"gpu {gpu} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    line = [l for l in proc.stdout.splitlines() if l.startswith("@@@")]
    if not line:
        raise SystemExit(f"gpu {gpu}: no result line\n{proc.stdout[-2000:]}")
    return json.loads(line[-1][3:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "02" / "timing-variance-amd.json")
    a = ap.parse_args()

    if 0 in a.gpus:
        raise SystemExit("GPU 0 is reserved for authoritative timing; use 1-7.")

    per_gpu = {}
    for g in a.gpus:
        print(f"  gpu {g} ...", flush=True)
        per_gpu[str(g)] = worker(g, a.reps)

    sizes = {}
    for size, nvidia in CASES:
        pooled, per = [], {}
        for g, res in per_gpu.items():
            r = res[str(size)]["ratios"]
            pooled += r
            per[g] = {"n": len(r), "median": statistics.median(r), "max": max(r)}
        # The test calls time_runnable ONCE per size, in a session where that
        # size's GEMM has not run before -- so what it samples is the *cold*
        # call, not the steady state. Split them, because pooling the two
        # answers a question nobody asked.
        cold = [res[str(size)]["ratios"][0] for res in per_gpu.values()]
        warm = [x for res in per_gpu.values() for x in res[str(size)]["ratios"][1:]]
        pooled.sort()
        observed_max = pooled[-1]
        # Round UP to the next 0.05 so the published constant is a clean number
        # that is never below max*MARGIN.
        threshold = math.ceil(observed_max * MARGIN / 0.05) * 0.05
        sizes[str(size)] = {
            "nvidia_threshold_replaced": nvidia,
            "n_samples": len(pooled),
            "median": round(statistics.median(pooled), 4),
            "p95": round(pooled[int(0.95 * (len(pooled) - 1))], 4),
            "p99": round(pooled[int(0.99 * (len(pooled) - 1))], 4),
            "observed_max": round(observed_max, 4),
            "amd_threshold": round(threshold, 4),
            "nvidia_threshold_would_fail_pct": round(
                100.0 * sum(1 for x in pooled if x >= nvidia) / len(pooled), 1),
            "cold_first_call": {"n": len(cold),
                                "values": [round(x, 3) for x in sorted(cold)],
                                "max": round(max(cold), 4)},
            "warm_steady_state": {"n": len(warm),
                                  "median": round(statistics.median(warm), 4),
                                  "p99": round(sorted(warm)[int(0.99 * (len(warm) - 1))], 4),
                                  "max": round(max(warm), 4)},
            "per_gpu": per,
            "raw_ratios": {g: [round(x, 4) for x in res[str(size)]["ratios"]]
                           for g, res in per_gpu.items()},
        }
        print(f"       cold(first call) max {max(cold):.2f}x | "
              f"warm median {statistics.median(warm):.2f}x max {max(warm):.2f}x")
        print(f"  mm[{size}]: median {statistics.median(pooled):.2f}x  "
              f"max {observed_max:.2f}x  -> threshold {threshold:.2f}x   "
              f"(NVIDIA {nvidia}x failed "
              f"{sizes[str(size)]['nvidia_threshold_would_fail_pct']}% of samples)")

    # test_variance_decreases_with_compute_intensity asserts CV(4096) < CV(64).
    cv64 = [c for r in per_gpu.values() for c in r["64"]["cvs"]]
    cv4k = [c for r in per_gpu.values() for c in r["4096"]["cvs"]]
    holds = sum(1 for s, l in zip(cv64, cv4k) if l < s)
    cv_check = {
        "rule": "CV(4096) < CV(64), paired per invocation",
        "n_pairs": len(cv64),
        "holds_pct": round(100.0 * holds / len(cv64), 1),
        "cv_64_median": round(statistics.median(cv64), 5),
        "cv_4096_median": round(statistics.median(cv4k), 5),
    }
    print(f"  CV(4096) < CV(64) holds in {cv_check['holds_pct']}% of "
          f"{cv_check['n_pairs']} paired invocations")

    from provenance import write_artifact
    write_artifact(a.out, "02-timing-variance-amd", {
        "_note": "AMD-derived replacements for the NVIDIA timing-variance "
                 "constants in tests/sol_execbench/core/bench/test_timing.py. "
                 "Same statistic as the test computes; only the constant is "
                 "re-derived. See scripts/derive_timing_variance.py.",
        "statistic": "max(times)/min(times) over one time_runnable(return_mode='all') call",
        "margin_over_observed_max": MARGIN,
        "gpus": a.gpus,
        "reps_per_gpu_per_size": a.reps,
        "sizes": sizes,
        "cv_monotonicity": cv_check,
    })
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
