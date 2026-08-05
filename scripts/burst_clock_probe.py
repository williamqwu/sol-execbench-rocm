#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does a short burst run at the same clock as a sustained loop? Go/no-go for
per-measurement F.

    python scripts/burst_clock_probe.py --out artifacts/01/burst-clock.json

Unlocked, a kernel's clock is set by its own power draw (`artifacts/01/unlocked-clock.json`:
1730 MHz for a dense GEMM at the 1400 W cap, 2394 MHz for a memory-bound one at
1170 W). To evaluate T_SOL at the clock a measurement ran at, we have to know that
clock -- and we cannot sample it during the measurement, because `time_runnable`
times `warmup=10 + rep=100` iterations of kernels that are mostly sub-millisecond.
That is a ~2 ms window against a 5 Hz sampler: zero samples, always.

The fallback is to characterize the clock in a separate sustained loop of the same
kernel. That is only valid if a short burst runs at the same clock as a sustained
loop. If short bursts boost higher -- which is physically plausible, since a burst
can finish before the power controller reacts -- then the characterization pass
understates F for exactly the short kernels, which loosens their bound and inflates
their score. A favourable bias that nothing downstream would reveal.

**No clock telemetry is used here, deliberately.** Per-iteration wall time at a
fixed shape is proportional to 1/clock, so if the clock is flat across burst lengths
the per-iteration time is flat too. That sidesteps the sampling-rate problem
entirely: the question "was the clock the same?" is answered by "did the work take
the same time per unit?", at whatever timescale we like.

Reading the output: `ratio` is per-iteration time at that burst length divided by
the longest (fully sustained) burst. A ratio below 1 means that burst ran FASTER per
iteration than sustained, i.e. it was boosting, i.e. characterizing on a sustained
loop would understate its clock.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Burst lengths in iterations. 110 is what time_runnable actually does
# (warmup 10 + rep 100); the rest bracket it by orders of magnitude up to a
# genuinely sustained loop.
BURSTS = [110, 400, 2_000, 10_000, 50_000]


def _shapes(dev, torch):
    """Representative of the corpus: a compute-bound GEMM, a small one, memory-bound."""
    out = {}

    a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    out["gemm_4096_compute_bound"] = (lambda: a @ b, 2.0 * 4096 ** 3)

    c = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
    d = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
    out["gemm_1024_small"] = (lambda: c @ d, 2.0 * 1024 ** 3)

    e = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    f = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    out["elementwise_memory_bound"] = (lambda: e + f, 0.0)

    return out


def time_burst(torch, dev, step, iters: int, repeats: int = 5) -> dict:
    """Median per-iteration time over *repeats* independent bursts.

    Each burst is preceded by an idle gap so the clock is not inherited from the
    previous burst -- that inheritance is the very effect being looked for, and
    letting it leak in would hide it.
    """
    per_iter = []
    for _ in range(repeats):
        time.sleep(0.35)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        torch.cuda.synchronize(dev)
        per_iter.append((time.perf_counter() - t0) / iters)
    return {
        "iters": iters,
        "per_iter_ms": statistics.median(per_iter) * 1e3,
        "per_iter_ms_min": min(per_iter) * 1e3,
        "per_iter_ms_max": max(per_iter) * 1e3,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--mode", choices=("unlocked", "locked"), default="unlocked",
                    help="locked applies setperfdeterminism, to separate a bias this "
                         "change would INTRODUCE from one already in our results")
    ap.add_argument("--setpoint", type=int, default=1660)
    ap.add_argument("--out", default="artifacts/01/burst-clock.json")
    a = ap.parse_args()

    if a.mode == "unlocked":
        subprocess.run(["rocm-smi", "--setperflevel", "auto"], capture_output=True)
    else:
        subprocess.run(["rocm-smi", "--setperfdeterminism", str(a.setpoint)],
                       capture_output=True)
    time.sleep(6)

    import torch
    dev = torch.device(f"cuda:{a.gpu}")
    shapes = _shapes(dev, torch)

    print(f"GPU {a.gpu}, {a.mode.upper()}. Per-iteration time vs burst length.")
    print("A short burst that is FASTER per iteration than a sustained loop was "
          "boosting,\nso characterizing its clock on a sustained loop would "
          "understate it.\n")

    results = {}
    verdict_worst = 0.0
    for name, (step, flops) in shapes.items():
        for _ in range(50):                 # compile / cache warm, outside timing
            step()
        torch.cuda.synchronize(dev)

        rows = [time_burst(torch, dev, step, n) for n in BURSTS]
        base = rows[-1]["per_iter_ms"]       # the sustained reference
        print(f"  {name}")
        print(f"    {'iters':>7} {'per-iter ms':>12} {'ratio':>7}  "
              f"{'implied clock vs sustained':>26}")
        for r in rows:
            ratio = r["per_iter_ms"] / base
            r["ratio_vs_sustained"] = ratio
            note = ""
            if r["iters"] == 110:
                note = "  <- what time_runnable does"
                verdict_worst = min(verdict_worst, ratio - 1.0)
            print(f"    {r['iters']:>7} {r['per_iter_ms']:>12.5f} {ratio:>7.3f}  "
                  f"{1/ratio:>25.3f}x{note}")
        if flops:
            print(f"    sustained throughput: "
                  f"{flops / (base * 1e-3) / 1e12:.1f} TFLOPS")
        results[name] = {"bursts": rows, "flops_per_iter": flops}
        print()

    print(f"VERDICT: at time_runnable's own burst length, per-iteration time "
          f"differs from\nsustained by at worst {100*verdict_worst:+.1f}%. "
          f"Positive/near-zero means a sustained\ncharacterization pass is a valid "
          f"stand-in; strongly negative means short bursts\nboost and the pass "
          f"would understate their clock.")

    from provenance import write_artifact
    write_artifact(a.out, "01-burst-clock", {
        "gpu": a.gpu, "mode": a.mode, "bursts": BURSTS, "results": results,
        "note": "Go/no-go for per-measurement F. Asks whether a short burst runs at "
                "the same clock as a sustained loop, using only wall-clock timing so "
                "the 5 Hz telemetry sampling rate is irrelevant.",
    })
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
