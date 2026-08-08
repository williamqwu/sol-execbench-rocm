#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Is the *unlocked* clock stable enough to time against? Three ways it could fail.

    python scripts/unlocked_clock_probe.py --out artifacts/01/unlocked-clock.json

`scripts/gpu_parity_check.py` established that with `perf_level=auto` all eight
GPUs are uniform to 3.4% in throughput, while `perf_determinism` spreads them 21%.
That invites an obvious question: skip the broken lock and time on all eight
unlocked.

The catch is *why* they were uniform. Unlocked, every card sat at 1377-1399 W
against a 1400 W cap. At the cap the clock is no longer an input, it is an output
of (power budget, how power-hungry this kernel is, temperature). The 3.4% figure
was measured with every card running the *same* GEMM -- an artificially uniform
condition that the real benchmark never has, since each card runs a different
problem.

So the clock could vary along three axes that a fixed lock would have removed, and
each is fatal to a different thing:

1. **Workload** -- a memory-bound kernel leaves the matrix cores idle, draws less
   power, and boosts higher than a dense GEMM. If this is large, absolute times are
   not comparable *between problems*, and worse, not comparable between the
   baseline and the candidate for the same problem.
2. **Time** -- 45 s cannot show coolant drift. A 20 h timing pass can. At the power
   cap, warmer means slower.
3. **Neighbours** -- what the other seven cards are doing changes this card's
   thermal and power environment.

Magnitudes decide the design, so this measures all three rather than reasoning
about them. Uses only torch and wall-clock timing.
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


def _mk_workload(kind: str, gpu: int):
    """Return (setup_fn, step_fn, flops_or_bytes_per_step, label)."""
    import torch
    dev = torch.device(f"cuda:{gpu}")

    if kind == "gemm_dense":
        a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
        b = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
        return lambda: (a @ b), 2.0 * 8192 ** 3, "flops"

    if kind == "gemm_small":
        a = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
        b = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
        return lambda: (a @ b), 2.0 * 1024 ** 3, "flops"

    if kind == "memory_bound":
        # 16384^2 bf16 = 512 MB each; a+b -> c touches 1.5 GB per step and leaves
        # the matrix cores idle, which is the low-power end of the spectrum.
        a = torch.randn(16384, 16384, device=dev, dtype=torch.bfloat16)
        b = torch.randn(16384, 16384, device=dev, dtype=torch.bfloat16)
        return lambda: (a + b), 3 * 16384 * 16384 * 2.0, "bytes"

    if kind == "reduction":
        a = torch.randn(16384, 16384, device=dev, dtype=torch.float32)
        return lambda: a.sum(), 16384 * 16384 * 4.0, "bytes"

    raise ValueError(kind)


def run_load(gpu: int, kind: str, seconds: float, result: dict) -> None:
    import torch
    step, per_step, unit = _mk_workload(kind, gpu)
    dev = torch.device(f"cuda:{gpu}")
    for _ in range(10):
        step()
    torch.cuda.synchronize(dev)

    n = 0
    t0 = time.perf_counter()
    deadline = t0 + seconds
    while time.perf_counter() < deadline:
        for _ in range(20):
            step()
        torch.cuda.synchronize(dev)
        n += 20
    dt = time.perf_counter() - t0
    result[gpu] = {
        "kind": kind, "steps": n, "seconds": dt,
        ("tflops" if unit == "flops" else "gbps"):
            n * per_step / dt / (1e12 if unit == "flops" else 1e9),
    }


def sample(gpus: list[int], stop: threading.Event, out: dict) -> None:
    from clock_calibrate import read_clocks
    while not stop.is_set():
        for g in gpus:
            c = read_clocks(g)
            if c.get("sclk_mhz"):
                c["t"] = time.time()
                out[g].append(c)
        time.sleep(0.5)


def probe(gpus: list[int], kind: str, seconds: float, label: str) -> dict:
    samples = {g: [] for g in gpus}
    results: dict = {}
    stop = threading.Event()
    s = threading.Thread(target=sample, args=(gpus, stop, samples), daemon=True)
    ws = [threading.Thread(target=run_load, args=(g, kind, seconds, results),
                           daemon=True) for g in gpus]
    s.start()
    for w in ws:
        w.start()
    for w in ws:
        w.join()
    stop.set()
    s.join(timeout=10)

    per = {}
    for g in gpus:
        # Drop the first 20% of samples: the clock takes a moment to respond to a
        # new load, and averaging the ramp in understates the steady state.
        sm = samples[g]
        sm = sm[len(sm) // 5:] or sm
        clk = [x["sclk_mhz"] for x in sm]
        pw = [x["power_w"] for x in sm if x.get("power_w")]
        tp = [x["temp_c"] for x in sm if x.get("temp_c")]
        r = results.get(g, {})
        per[g] = {
            "clock_median_mhz": statistics.median(clk) if clk else None,
            "clock_p05_mhz": min(clk) if clk else None,
            "clock_p95_mhz": max(clk) if clk else None,
            "power_median_w": statistics.median(pw) if pw else None,
            "temp_median_c": statistics.median(tp) if tp else None,
            "throughput": r.get("tflops") or r.get("gbps"),
            "throughput_unit": "TFLOPS" if "tflops" in r else "GB/s",
            "clock_series": clk,
        }
    return {"label": label, "kind": kind, "gpus": gpus, "per_gpu": per}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1, help="card for the workload sweep")
    ap.add_argument("--n-gpus", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--drift-seconds", type=float, default=480.0)
    ap.add_argument("--out", default="artifacts/01/unlocked-clock.json")
    a = ap.parse_args()

    subprocess.run(["rocm-smi", "--setperflevel", "auto"], capture_output=True)
    time.sleep(5)

    blocks = []
    kinds = ["gemm_dense", "gemm_small", "memory_bound", "reduction"]

    print(f"=== axis 1: does the UNLOCKED clock depend on the workload? "
          f"(GPU {a.gpu} alone) ===")
    print("  workload       clock    range      power   throughput")
    for k in kinds:
        b = probe([a.gpu], k, a.seconds, f"workload {k}, GPU {a.gpu} alone")
        blocks.append(b)
        r = b["per_gpu"][a.gpu]
        print(f"  {k:<13} {r['clock_median_mhz']:>6.0f}  "
              f"{r['clock_p05_mhz']:>4.0f}-{r['clock_p95_mhz']:<4.0f} "
              f"{r['power_median_w']:>6.0f} W  "
              f"{r['throughput']:>8.1f} {r['throughput_unit']}")
    cl = [b["per_gpu"][a.gpu]["clock_median_mhz"] for b in blocks]
    print(f"  --> clock varies {min(cl):.0f}-{max(cl):.0f} MHz across workloads "
          f"= {100*(max(cl)-min(cl))/max(cl):.1f}%")

    print(f"\n=== axis 3: does a card's clock depend on its NEIGHBOURS? ===")
    gpus = list(range(a.n_gpus))
    solo = blocks[0]["per_gpu"][a.gpu]["clock_median_mhz"]
    b_all = probe(gpus, "gemm_dense", a.seconds, "gemm_dense, all 8 loaded")
    blocks.append(b_all)
    together = b_all["per_gpu"][a.gpu]["clock_median_mhz"]
    print(f"  GPU {a.gpu} on gemm_dense: {solo:.0f} MHz alone -> "
          f"{together:.0f} MHz with 7 busy neighbours "
          f"({100*(together-solo)/solo:+.1f}%)")
    allc = [b_all["per_gpu"][g]["clock_median_mhz"] for g in gpus]
    print(f"  all 8 clocks: {min(allc):.0f}-{max(allc):.0f} MHz "
          f"({100*(max(allc)-min(allc))/max(allc):.1f}% spread)")

    print(f"\n=== axis 2: does it DRIFT? all 8 on gemm_dense for "
          f"{a.drift_seconds/60:.0f} min ===")
    b_d = probe(gpus, "gemm_dense", a.drift_seconds, "drift, all 8 loaded")
    blocks.append(b_d)
    print("  gpu   first30s   last30s    delta    temp")
    worst = 0.0
    for g in gpus:
        ser = b_d["per_gpu"][g]["clock_series"]
        if len(ser) < 20:
            continue
        head = statistics.median(ser[:60])
        tail = statistics.median(ser[-60:])
        d = 100 * (tail - head) / head
        worst = min(worst, d)
        print(f"  {g:>3}  {head:>8.0f}  {tail:>8.0f}  {d:>+6.1f}%  "
              f"{b_d['per_gpu'][g]['temp_median_c']:>4.0f} C")
    print(f"  --> worst drift {worst:+.1f}%")

    from provenance import write_artifact
    write_artifact(a.out, "01-unlocked-clock", {
        "blocks": blocks,
        "note": "Can we time on all 8 GPUs unlocked? Measures the three ways an "
                "unlocked clock can move that a working lock would have pinned: "
                "workload dependence, drift over time, neighbour coupling.",
    })
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
