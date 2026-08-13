#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One problem's reference kernel, timed with the clock and power it drew.

Runs INSIDE the measurement container, on whatever single card
``HIP_VISIBLE_DEVICES`` exposes. It does not touch the clock policy -- the host
driver (`scripts/clock_ab.py`) owns that, because only the host has sudo and
only one process may be allowed to restore the lock.

    HIP_VISIBLE_DEVICES=0 python /work/scripts/bounds/clock_ab_probe.py \
        --problem L1__074_fused_gated_mlp_silu --reps 7 --hold 4 --out /tmp/x.json

Reports, per rep: wall-clock ms per call, mean GFX clock, mean power. The rep
is the unit of variance -- a single mean over one long hold cannot tell a card
that is steady from one that alternates between two clocks.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "/work/scripts")
sys.path.insert(0, "/work/scripts/runners")

BASE = Path("/work/data/SOL-ExecBench/benchmark")


def _sampler(handle, stop, clocks, powers, mem_clocks):
    import amdsmi

    while not stop.is_set():
        try:
            c = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX)
            v = c.get("clk") or c.get("cur_clk")
            if v:
                clocks.append(v)
            m = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.MEM)
            mv = m.get("clk") or m.get("cur_clk")
            if mv:
                mem_clocks.append(mv)
            p = amdsmi.amdsmi_get_power_info(handle)
            pv = p.get("current_socket_power") or p.get("average_socket_power")
            if pv:
                powers.append(pv)
        except Exception:  # noqa: BLE001 -- a dropped sample is not a failure
            pass
        time.sleep(0.002)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True,
                    help="comma-separated keys like L1__074_fused_gated_mlp_silu. "
                         "Several in one process because torch startup and the "
                         "container cost more than the measurement does.")
    ap.add_argument("--workload", type=int, default=-1,
                    help="workload index; -1 picks the largest by axes")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--hold", type=float, default=4.0,
                    help="seconds of sustained load per rep")
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    import torch
    import amdsmi

    from gpu_map import torch_to_amdsmi
    from _common import exec_reference, load_problem, prepare_inputs

    amdsmi.amdsmi_init()
    handle = amdsmi.amdsmi_get_processor_handles()[torch_to_amdsmi()[0]]

    keys = [k for k in a.problem.split(",") if k]
    out_all = {}
    for key in keys:
        try:
            out_all[key] = measure(key, a, torch, handle)
        except Exception as exc:  # noqa: BLE001 -- one bad problem is a row,
            # not the end of the run: the other problems are still evidence.
            out_all[key] = {"ok": False,
                            "error": f"{type(exc).__name__}: {exc}"[:400]}
        print(json.dumps({key: {k: v for k, v in out_all[key].items()
                                if k != "reps"}}), flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out_all, indent=1))
    amdsmi.amdsmi_shut_down()
    return 0


def _synthetic(key, torch):
    """A saturating GEMM, as the calibration load rather than as a problem.

    The benchmark's own kernels turn out not to reach the power cap, so a
    lock-vs-no-lock comparison over them can only ever say "the cap never
    bound". The load F_LOCK was calibrated against -- a dense bf16 GEMM that
    does hit the cap -- is the one case where the setpoint is the binding
    constraint, so it has to be in the comparison explicitly.
    """
    dtype, n = {
        "synthetic__bf16_gemm_8192": (torch.bfloat16, 8192),
        "synthetic__fp32_gemm_8192": (torch.float32, 8192),
        "synthetic__fp16_gemm_8192": (torch.float16, 8192),
    }[key]
    x = torch.randn(n, n, device="cuda:0", dtype=dtype)
    y = torch.randn(n, n, device="cuda:0", dtype=dtype)
    return (lambda: torch.mm(x, y)), [str(dtype)], (x, y)


def measure(key, a, torch, handle) -> dict:
    from _common import exec_reference, load_problem, prepare_inputs

    if key.startswith("synthetic__"):
        return _measure_call(key, a, torch, handle, *_synthetic(key, torch),
                             widx=-1, uuid=None)

    cat, name = key.split("__", 1)
    definition, workloads = load_problem(BASE / cat / name)
    run_ref, ns = exec_reference(definition)
    if a.workload < 0:
        w = max(workloads, key=lambda x: json.dumps(getattr(x, "axes", {}) or {}))
        widx = workloads.index(w)
    else:
        widx = a.workload
        w = workloads[widx]
    inputs = prepare_inputs(definition, w, ns)
    call = ((lambda: run_ref(**inputs)) if isinstance(inputs, dict)
            else (lambda: run_ref(*inputs)))
    dtypes = sorted({str(v.dtype) for v in
                     (inputs.values() if isinstance(inputs, dict) else inputs)
                     if hasattr(v, "dtype")})
    return _measure_call(key, a, torch, handle, call, dtypes, inputs,
                         widx=widx, uuid=getattr(w, "uuid", None))


def _measure_call(key, a, torch, handle, call, dtypes, held, widx, uuid) -> dict:
    # Warm up until the card has settled: a cold card clocks high for the first
    # second whatever the policy, which is exactly the confound this test is
    # about.
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < a.warmup_s:
        for _ in range(20):
            call()
        torch.cuda.synchronize()

    reps = []
    for _ in range(a.reps):
        clocks: list[int] = []
        powers: list[float] = []
        mem_clocks: list[int] = []
        stop = threading.Event()
        th = threading.Thread(target=_sampler,
                              args=(handle, stop, clocks, powers, mem_clocks),
                              daemon=True)
        torch.cuda.synchronize()
        th.start()
        t0 = time.perf_counter()
        n = 0
        # Sync every 50 so the hold is wall time and not a queue backlog.
        while time.perf_counter() - t0 < a.hold:
            for _ in range(50):
                call()
            torch.cuda.synchronize()
            n += 50
        elapsed = time.perf_counter() - t0
        stop.set()
        th.join(timeout=2)
        reps.append({
            "calls": n,
            "elapsed_s": elapsed,
            "ms_per_call": elapsed * 1000.0 / n,
            "gfx_mhz_mean": statistics.mean(clocks) if clocks else None,
            "gfx_mhz_p50": statistics.median(clocks) if clocks else None,
            "gfx_mhz_max": max(clocks) if clocks else None,
            "gfx_mhz_min": min(clocks) if clocks else None,
            "gfx_samples": len(clocks),
            "mem_mhz_p50": statistics.median(mem_clocks) if mem_clocks else None,
            "power_w_mean": statistics.mean(powers) if powers else None,
            "power_w_max": max(powers) if powers else None,
        })

    ms = [r["ms_per_call"] for r in reps]
    payload = {
        "ok": True,
        "problem": key,
        "workload_index": widx,
        "workload_uuid": uuid,
        "input_dtypes": dtypes,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "torch_device_name": torch.cuda.get_device_name(0),
        "reps": reps,
        "ms_per_call_p50": statistics.median(ms),
        "ms_per_call_mean": statistics.mean(ms),
        "ms_per_call_cv": (statistics.pstdev(ms) / statistics.mean(ms)) if len(ms) > 1 else 0.0,
        "gfx_mhz_p50_of_reps": statistics.median(
            [r["gfx_mhz_p50"] for r in reps if r["gfx_mhz_p50"]] or [0]) or None,
        "power_w_mean_of_reps": statistics.mean(
            [r["power_w_mean"] for r in reps if r["power_w_mean"]] or [0]) or None,
    }
    # Problems in one process share a card's memory; a big one left resident
    # can OOM the next.
    del held, call
    torch.cuda.empty_cache()
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
