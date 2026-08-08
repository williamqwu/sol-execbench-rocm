#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What does a SHORT timed window cost, and is the clock responsible?

    python scripts/burst_clock_probe.py --out artifacts/01/burst-clock.json
    python scripts/burst_clock_probe.py --mode locked --setpoint 1660 --out ...

`BenchmarkConfig` defaults to `warmup_runs=10, iterations=50`, so `time_runnable`
times 60 back-to-back executions. For the sub-millisecond kernels that make up most
of this corpus that is a 1-13 ms window -- shorter than any telemetry sampler can
observe, which is why "what clock was this measured at?" has only ever been assumed
for an artifact, never established.

**No clock telemetry is used here, deliberately**, and that is the whole trick. At a
fixed shape, per-iteration wall time is proportional to 1/clock, so timing the same
kernel at increasing burst lengths converts "was the clock the same?" into "did the
work take the same time per unit?" -- a question answerable at any timescale, which
sidesteps the sampling-rate problem instead of fighting it.

Each burst is preceded by an idle gap, so a burst cannot inherit whatever state the
previous one left warm. Run `--mode locked` as well: an effect that is present under
`perf_determinism` too is a property of the timing window rather than of a node that
cannot pin its clock.

Reading the output. `ratio` is per-iteration time at that burst length over the
longest (fully sustained) burst. Above 1 means the short burst was SLOWER per
iteration. The absolute delta matters more than the ratio for attribution: a
depressed clock slows every kernel roughly in proportion, so if one kernel is immune
while others pay tens of microseconds, the clock is not the mechanism. That is what
the memory-bound shape is here for -- it is the control, not a third data point.
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

# Burst lengths in iterations. 60 is what time_runnable actually does under the
# shipped BenchmarkConfig (warmup_runs 10 + iterations 50); the rest bracket it by
# orders of magnitude up to a genuinely sustained loop. Keep the first entry equal
# to the real default -- an earlier version used 110 on the assumption of rep=100,
# which understated the effect, since a shorter window can only cost more.
BURSTS = [60, 400, 2_000, 10_000, 50_000]


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

    print(f"GPU {a.gpu}, {a.mode.upper()}. Per-iteration time vs burst length.\n")

    results = {}
    at_default: dict[str, tuple[float, float]] = {}     # name -> (ratio, delta_us)
    for name, (step, flops) in shapes.items():
        for _ in range(50):                 # compile / cache warm, outside timing
            step()
        torch.cuda.synchronize(dev)

        rows = [time_burst(torch, dev, step, n) for n in BURSTS]
        base = rows[-1]["per_iter_ms"]       # the sustained reference
        print(f"  {name}")
        print(f"    {'iters':>7} {'per-iter ms':>12} {'ratio':>7} {'Δ vs sustained':>16}")
        for r in rows:
            ratio = r["per_iter_ms"] / base
            delta_us = (r["per_iter_ms"] - base) * 1e3
            r["ratio_vs_sustained"] = ratio
            r["delta_vs_sustained_us"] = delta_us
            note = ""
            if r["iters"] == BURSTS[0]:
                note = "  <- what time_runnable does"
                at_default[name] = (ratio, delta_us)
            print(f"    {r['iters']:>7} {r['per_iter_ms']:>12.5f} {ratio:>7.3f} "
                  f"{delta_us:>14.1f}µs{note}")
        if flops:
            print(f"    sustained throughput: "
                  f"{flops / (base * 1e-3) / 1e12:.1f} TFLOPS")
        results[name] = {"bursts": rows, "flops_per_iter": flops}
        print()

    worst = max(at_default.values(), key=lambda x: x[0]) if at_default else (1.0, 0.0)
    print(f"At time_runnable's own burst length the worst shape reads "
          f"{100 * (worst[0] - 1):+.1f}% vs sustained.\n")

    # The ATTRIBUTION, which is the point of running more than one shape. A depressed
    # clock slows everything roughly in proportion, so proportional deltas implicate
    # the clock and concentrated ones implicate whatever path the affected shapes
    # share. Reported rather than concluded: this prints the spread and names the
    # reading, it does not decide for the reader.
    deltas = {n: d for n, (_, d) in at_default.items()}
    if deltas and max(deltas.values()) > 0:
        lo, hi = min(deltas.values()), max(deltas.values())
        print("  Δ per iteration by shape (the attribution):")
        for n, d in sorted(deltas.items(), key=lambda kv: -kv[1]):
            print(f"    {n:<30} {d:>7.1f}µs")
        if lo <= 0.1 * hi:
            print(f"\n  Spread is {hi / max(lo, 1e-9):.0f}x across shapes, so the cost is "
                  f"NOT proportional:\n  a depressed clock would slow all of them "
                  f"alike. Look at what the expensive\n  shapes share and the cheap "
                  f"one does not.")
        else:
            print("\n  Cost is roughly proportional across shapes, which is what a "
                  "clock effect\n  looks like. Worth checking against a direct "
                  "in-loop clock sample.")

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
