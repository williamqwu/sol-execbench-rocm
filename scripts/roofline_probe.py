#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure achieved rooflines: HBM bandwidth, BF16 GEMM, and LLC bandwidth.

These are the *empirical* ceilings published alongside the analytic SOL bounds
(docs/plan-2026-07-31.md §7.1), so a reader can see what fraction of theoretical peak is
actually reachable on this hardware. That is how "analytic peaks are reachable
to different degrees on different microarchitectures" gets handled honestly
instead of by quietly forking the methodology.

    python scripts/roofline_probe.py --gpu 0 --out artifacts/00/roofline-gpu0.json
    python scripts/roofline_probe.py --gpu 0 --llc-sweep   # task 03 V2 + task 02 flush check

Peaks to compare against are looked up per part (``solexbench_rocm.parts``),
never hardcoded: MI350X and MI355X are the same die at different clocks, so a
fixed "2500 TFLOPS" denominator would understate MI350X's achieved fraction by
9% while looking perfectly reasonable.

Task 00 runs this at default clocks (reference points only — do NOT cite them
downstream). Re-run at F_LOCK after task 01 for the numbers that go in the
manifest.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from provenance import write_artifact  # noqa: E402
from solexbench_rocm.parts import detect_part  # noqa: E402


def _sync(dev):
    import torch
    torch.cuda.synchronize(dev)


def hbm_bandwidth(gpu: int, gib: int = 16, iters: int = 30) -> dict:
    """Achieved HBM bandwidth via large device-to-device copy.

    Buffer must far exceed the 256 MB LLC or this measures cache, not HBM.
    """
    import torch
    dev = torch.device(f"cuda:{gpu}")
    n = gib * 2**30 // 2                       # fp16 elements
    src = torch.empty(n, dtype=torch.float16, device=dev)
    dst = torch.empty_like(src)
    nbytes = src.numel() * src.element_size()

    for _ in range(5):
        dst.copy_(src)
    _sync(dev)

    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); dst.copy_(src); e.record()
        _sync(dev)
        times.append(s.elapsed_time(e) / 1e3)

    t = statistics.median(times)
    return {"buffer_gib": gib,
            "median_s": t,
            # copy moves the buffer twice: one read + one write
            "tbs": (2 * nbytes) / t / 1e12,
            "note": "read+write counted; buffer >> LLC so this is HBM"}


def gemm_bf16(gpu: int, size: int = 8192, iters: int = 30) -> dict:
    import torch
    dev = torch.device(f"cuda:{gpu}")
    a = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    for _ in range(10):
        a @ b
    _sync(dev)

    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); a @ b; e.record()
        _sync(dev)
        times.append(s.elapsed_time(e) / 1e3)

    t = statistics.median(times)
    return {"size": size, "median_s": t,
            "tflops": (2 * size**3) / t / 1e12}


def llc_sweep(gpu: int) -> dict:
    """Bandwidth vs working-set size. The cliff locates the real LLC capacity.

    Serves two purposes:
      * task 03 V2 — Infinity Cache bandwidth, a SOLAR arch-config input
      * task 02    — validates the 512 MB flush-buffer sizing. If the cliff is
                     not near 256 MB, either the LLC number or the flush
                     mechanism is wrong. Find out which before proceeding.
    """
    import torch
    dev = torch.device(f"cuda:{gpu}")
    points = []
    for mib in (8, 16, 32, 64, 128, 192, 256, 384, 512, 1024, 2048):
        n = mib * 2**20 // 2
        src = torch.empty(n, dtype=torch.float16, device=dev)
        dst = torch.empty_like(src)
        for _ in range(10):
            dst.copy_(src)
        _sync(dev)
        times = []
        for _ in range(50):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); dst.copy_(src); e.record()
            _sync(dev)
            times.append(s.elapsed_time(e) / 1e3)
        t = statistics.median(times)
        points.append({"mib": mib,
                       "tbs": (2 * src.numel() * 2) / t / 1e12})
        del src, dst
        torch.cuda.empty_cache()

    cliff = None
    for prev, cur in zip(points, points[1:]):
        if prev["tbs"] > 0 and cur["tbs"] < prev["tbs"] * 0.75:
            cliff = {"between_mib": [prev["mib"], cur["mib"]],
                     "drop": 1 - cur["tbs"] / prev["tbs"]}
            break

    return {"points": points, "cliff": cliff,
            "expected_llc_mib": 256,
            "interpretation":
                "cliff near 256 MiB confirms Infinity Cache capacity and "
                "validates the 512 MB flush buffer; elsewhere means the LLC "
                "number or the flush mechanism is wrong"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="artifacts/00/roofline.json")
    ap.add_argument("--llc-sweep", action="store_true")
    a = ap.parse_args()

    part = detect_part(a.gpu)
    payload: dict = {
        "gpu": a.gpu,
        # Which part these ceilings belong to, and the peaks they are a
        # fraction OF. Recorded in the artifact so a reader never has to infer
        # the denominator, and so an MI350X number can never be read against an
        # MI355X peak.
        "part": part.name,
        "part_peak": {
            "freq_ghz": part.peak_freq_ghz,
            "power_cap_w": part.power_cap_w,
            "hbm_tbs": part.dram_bytes_per_sec / 1e12,
            "bf16_tflops": part.peak_flops("bf16_tc") / 1e12,
        },
    }
    for name, fn in (("hbm", lambda: hbm_bandwidth(a.gpu)),
                     ("gemm", lambda: gemm_bf16(a.gpu))):
        try:
            payload[name] = fn()
        except Exception as e:
            payload[name] = {"error": str(e)}    # record, never guess

    payload["hbm_tbs"] = payload.get("hbm", {}).get("tbs")
    payload["gemm_bf16_tflops"] = payload.get("gemm", {}).get("tflops")

    if a.llc_sweep:
        try:
            payload["llc_sweep"] = llc_sweep(a.gpu)
        except Exception as e:
            payload["llc_sweep"] = {"error": str(e)}

    peak_hbm = part.dram_bytes_per_sec / 1e12
    peak_gemm = part.peak_flops("bf16_tc") / 1e12
    for key, got, peak, unit in (
        ("hbm_tbs", payload["hbm_tbs"], peak_hbm, "TB/s"),
        ("gemm_bf16_tflops", payload["gemm_bf16_tflops"], peak_gemm, "TFLOPS BF16"),
    ):
        if isinstance(got, (int, float)):
            payload[f"{key}_frac_of_peak"] = got / peak

    write_artifact(a.out, "roofline-probe", payload)
    print(f"part {part.name}  (peak {part.peak_freq_ghz} GHz, {part.power_cap_w} W)")
    print(f"HBM  {payload['hbm_tbs']} TB/s   "
          f"({payload.get('hbm_tbs_frac_of_peak', float('nan')):.1%} of {peak_hbm:.1f} spec peak)")
    print(f"GEMM {payload['gemm_bf16_tflops']} TFLOPS BF16   "
          f"({payload.get('gemm_bf16_tflops_frac_of_peak', float('nan')):.1%} of "
          f"{peak_gemm:.0f} spec peak @ {part.peak_freq_ghz} GHz)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
