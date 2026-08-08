#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Do all eight GPUs do the same work at the same speed? A control experiment.

    python scripts/gpu_parity_check.py --out artifacts/00/gpu-parity.json

Exists to answer a specific doubt about the node report in
``artifacts/00/node-defect-report.md``: **is the clock disparity real, or an artifact
of how we measure it?** Three controls, because the answer has to survive all of
them.

1. **Identical work, and throughput measured independently of the clock.** Every
   GPU runs the same fixed-shape BF16 GEMM the same number of times. Wall time gives
   TFLOPS. If a card reporting 80% of the clock also delivers 80% of the FLOPS, the
   clock telemetry is corroborated by work actually done -- so it is not a sensor
   lying, and not our sampling.

2. **Lock off as well as on.** If the cards differ with `perf_level=auto`, the
   hardware differs. If they only differ under `perf_determinism`, the lock
   mechanism is what misbehaves. These are different tickets.

3. **Alone as well as together.** If a card is only slow while its neighbours work,
   that is contention or a shared budget. If it is slow alone, it is the card.

Nothing here uses the benchmark's own timing harness, tolerances, or bound
machinery -- just torch, a GEMM, and a clock sampler -- so a defect in those cannot
produce the result.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

SIZE = 8192
ITERS = 400


def _flops_per_iter(size: int) -> float:
    # One GEMM: 2*N^3 flops.
    return 2.0 * size ** 3


def sample_clocks(gpus: list[int], stop: threading.Event,
                  out: dict[int, list[dict]]) -> None:
    from clock_calibrate import read_clocks

    while not stop.is_set():
        for g in gpus:
            c = read_clocks(g)
            if c.get("sclk_mhz"):
                out[g].append(c)
        time.sleep(0.5)


def run_gemm(gpu: int, size: int, seconds: float, result: dict) -> None:
    """Fixed shape, no autotuning, run for a fixed DURATION.

    Duration rather than a fixed iteration count, because a count tuned to finish
    quickly makes the timed region shorter than the clock sampler's interval: the
    first version of this ran 300 iterations, which at ~1400 TFLOPS is 0.24 s, and
    the sampler then reported 158 MHz because it was mostly reading an idle GPU.
    The throughput numbers were still sound -- wall clock does not care -- but the
    clock column was meaningless. A clock has to be sampled across a load that is
    actually sustained.
    """
    import torch

    dev = torch.device(f"cuda:{gpu}")
    a = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    for _ in range(20):                      # warm up outside the timed region
        a @ b
    torch.cuda.synchronize(dev)

    iters = 0
    t0 = time.perf_counter()
    deadline = t0 + seconds
    while time.perf_counter() < deadline:
        for _ in range(50):
            a @ b
        torch.cuda.synchronize(dev)
        iters += 50
    dt = time.perf_counter() - t0

    result[gpu] = {
        "seconds": dt,
        "tflops": iters * _flops_per_iter(size) / dt / 1e12,
        "iters": iters,
        "size": size,
    }


def measure(gpus: list[int], size: int, seconds: float, label: str) -> dict:
    from clock_calibrate import read_clocks  # noqa: F401  (import check)

    samples: dict[int, list[dict]] = {g: [] for g in gpus}
    results: dict[int, dict] = {}
    stop = threading.Event()
    sampler = threading.Thread(target=sample_clocks, args=(gpus, stop, samples),
                               daemon=True)
    workers = [threading.Thread(target=run_gemm, args=(g, size, seconds, results),
                               daemon=True) for g in gpus]
    sampler.start()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    stop.set()
    sampler.join(timeout=10)

    rows = {}
    for g in gpus:
        s = samples[g]
        clk = [x["sclk_mhz"] for x in s]
        pw = [x.get("power_w") for x in s if x.get("power_w")]
        tp = [x.get("temp_c") for x in s if x.get("temp_c")]
        r = results.get(g, {})
        rows[g] = {
            "tflops": r.get("tflops"),
            "seconds": r.get("seconds"),
            "clock_median_mhz": statistics.median(clk) if clk else None,
            "clock_min_mhz": min(clk) if clk else None,
            "power_median_w": statistics.median(pw) if pw else None,
            "temp_median_c": statistics.median(tp) if tp else None,
            "n_clock_samples": len(clk),
        }
    return {"label": label, "gpus": gpus, "per_gpu": rows}


def show(block: dict) -> None:
    print(f"\n--- {block['label']} ---")
    print("  gpu   TFLOPS   clock    power    temp   TFLOPS/GHz")
    ref = None
    for g in block["gpus"]:
        r = block["per_gpu"][g]
        tf, clk = r["tflops"], r["clock_median_mhz"]
        eff = (tf / (clk / 1000)) if (tf and clk) else None
        if ref is None and tf:
            ref = tf
        print(f"  {g:>3}  {tf:>7.1f}  {clk:>6.0f}  {r['power_median_w']:>6.0f} W  "
              f"{r['temp_median_c']:>4.0f} C  {eff:>9.1f}"
              if tf and clk else f"  {g:>3}  (no result)")
    tfs = [block["per_gpu"][g]["tflops"] for g in block["gpus"]
           if block["per_gpu"][g]["tflops"]]
    clks = [block["per_gpu"][g]["clock_median_mhz"] for g in block["gpus"]
            if block["per_gpu"][g]["clock_median_mhz"]]
    if tfs and clks:
        print(f"  TFLOPS spread {min(tfs):.1f}-{max(tfs):.1f} "
              f"({100*(max(tfs)-min(tfs))/max(tfs):.1f}%)   "
              f"clock spread {min(clks):.0f}-{max(clks):.0f} "
              f"({100*(max(clks)-min(clks))/max(clks):.1f}%)")


def set_level(level: str, setpoint: int | None = None) -> None:
    if level == "auto":
        subprocess.run(["rocm-smi", "--setperflevel", "auto"],
                       capture_output=True, text=True)
    else:
        subprocess.run(["rocm-smi", "--setperfdeterminism", str(setpoint)],
                       capture_output=True, text=True)
    time.sleep(5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-gpus", type=int, default=8)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--seconds", type=float, default=45.0,
                    help="duration of each load, long enough that the "
                         "clock sampler sees sustained load")
    ap.add_argument("--setpoint", type=int, default=1660)
    ap.add_argument("--out", default="artifacts/00/gpu-parity.json")
    a = ap.parse_args()

    gpus = list(range(a.n_gpus))
    blocks = []

    print(f"identical work on every GPU: BF16 GEMM {a.size}^3 for {a.seconds}s")
    print("TFLOPS/GHz is the control: if a slow card's clock is real, its "
          "throughput\nper unit clock should match the others.")

    set_level("auto")
    blocks.append(measure(gpus, a.size, a.seconds, "UNLOCKED (perf_level auto), all together"))
    show(blocks[-1])

    set_level("determinism", a.setpoint)
    blocks.append(measure(gpus, a.size, a.seconds,
                          f"LOCKED (setperfdeterminism {a.setpoint}), all together"))
    show(blocks[-1])

    for g in (0, 1, 2):
        if g < a.n_gpus:
            blocks.append(measure([g], a.size, a.seconds,
                                  f"LOCKED {a.setpoint}, GPU {g} ALONE"))
            show(blocks[-1])

    from provenance import write_artifact

    write_artifact(a.out, "00-gpu-parity", {
        "size": a.size, "seconds": a.seconds, "setpoint_mhz": a.setpoint,
        "blocks": blocks,
        "note": "Control experiment for artifacts/00/node-defect-report.md. Identical "
                "fixed-shape work on every GPU, throughput measured by wall clock so "
                "it is independent of the clock telemetry, run unlocked and locked "
                "and both together and alone. Uses no part of the benchmark's timing "
                "harness.",
    })
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
