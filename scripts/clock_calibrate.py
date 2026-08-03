#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 01 — clock calibration for MI355X.

    floor         sustained-load clock floor  -> the basis for choosing F_LOCK
    lock          apply deterministic clocks
    verify        confirm the lock holds UNDER LOAD (unloaded checks lie)
    stability     timing reproducibility at F_LOCK  (gate: CV < 2%)
    interference  does sibling-GPU load perturb timing?  (schedule-shaping)

!! NOT YET RUN ON HARDWARE. Structure and logic are reviewed; exact amd-smi
   field names and the sysfs fallback paths are the most likely things to need
   a small fix on first contact. Fix them, then record in STATE.md that you did
   -- a later session will otherwise wonder whether the numbers predate the fix.

Never guess F_LOCK. See tasks/01-clock-calibration.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import write_artifact  # noqa: E402

SAMPLE_HZ = 1.0
STABILITY_CV_GATE = 0.02


# --------------------------------------------------------------------------
# SMI access. amdsmi python lib preferred; rocm-smi subprocess as fallback.
# --------------------------------------------------------------------------

def _amdsmi():
    try:
        import amdsmi
        amdsmi.amdsmi_init()
        return amdsmi
    except Exception:
        return None


def read_clocks(gpu: int) -> dict:
    """Return {sclk_mhz, mclk_mhz, power_w, temp_c, throttle} for *gpu*."""
    smi = _amdsmi()
    if smi is not None:
        try:
            handle = smi.amdsmi_get_processor_handles()[gpu]
            sclk = smi.amdsmi_get_clock_info(handle, smi.AmdSmiClkType.GFX)
            mclk = smi.amdsmi_get_clock_info(handle, smi.AmdSmiClkType.MEM)
            power = smi.amdsmi_get_power_info(handle)
            temp = smi.amdsmi_get_temp_metric(
                handle, smi.AmdSmiTemperatureType.EDGE,
                smi.AmdSmiTemperatureMetric.CURRENT)
            return {
                "sclk_mhz": sclk.get("clk"), "mclk_mhz": mclk.get("clk"),
                "power_w": power.get("current_socket_power"),
                "temp_c": temp, "source": "amdsmi",
            }
        except Exception as e:
            return {"error": f"amdsmi: {e}", "source": "amdsmi"}

    out = subprocess.run(
        ["rocm-smi", "-d", str(gpu), "--showgpuclocks", "--showpower",
         "--showtemp", "--json"],
        capture_output=True, text=True)
    if out.returncode != 0:
        return {"error": out.stderr.strip(), "source": "rocm-smi"}
    try:
        return {"raw": json.loads(out.stdout), "source": "rocm-smi"}
    except json.JSONDecodeError:
        return {"error": "unparseable rocm-smi output", "source": "rocm-smi"}


def set_perf_determinism(freq_mhz: int, gpu: int | None = None) -> bool:
    """AMD's documented determinism mechanism: cap the soft max clock."""
    cmd = ["rocm-smi", "--setperfdeterminism", str(freq_mhz)]
    if gpu is not None:
        cmd += ["-d", str(gpu)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(f"  FAILED: {out.stderr.strip()}", file=sys.stderr)
        return False
    return True


def reset_clocks() -> None:
    subprocess.run(["rocm-smi", "-r"], capture_output=True)


# --------------------------------------------------------------------------
# Load generation
# --------------------------------------------------------------------------

def _sustained_load(gpu: int, seconds: float, size: int = 8192):
    """Saturate the matrix cores with back-to-back BF16 GEMMs."""
    import torch
    dev = torch.device(f"cuda:{gpu}")
    a = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    deadline = time.time() + seconds
    while time.time() < deadline:
        for _ in range(20):
            a @ b
        torch.cuda.synchronize(dev)


def _timed_reference(gpu: int, size: int = 4096, iters: int = 50) -> float:
    """Median ms of a fixed GEMM. The stability/interference probe."""
    import torch
    dev = torch.device(f"cuda:{gpu}")
    a = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    for _ in range(10):
        a @ b
    torch.cuda.synchronize(dev)
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        a @ b
        e.record()
        torch.cuda.synchronize(dev)
        times.append(s.elapsed_time(e))
    return statistics.median(times)


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------

def cmd_floor(args):
    """Sustained-load clock floor. p5 of the FINAL 5 MINUTES, not the ramp."""
    import threading
    total = args.minutes * 60
    stop = threading.Event()

    def load():
        try:
            _sustained_load(args.gpu, total)
        except Exception as e:
            print(f"load thread died: {e}", file=sys.stderr)
        finally:
            stop.set()

    t = threading.Thread(target=load, daemon=True)
    t.start()

    samples = []
    start = time.time()
    while time.time() - start < total and not stop.is_set():
        s = read_clocks(args.gpu)
        s["t"] = time.time() - start
        samples.append(s)
        time.sleep(1.0 / SAMPLE_HZ)

    tail_start = max(0, total - 300)
    tail = [s["sclk_mhz"] for s in samples
            if s.get("t", 0) >= tail_start and s.get("sclk_mhz")]

    result = {"gpu": args.gpu, "minutes": args.minutes,
              "n_samples": len(samples), "n_tail": len(tail), "samples": samples}
    if tail:
        tail.sort()
        result["steady_state"] = {
            "p5_mhz": tail[max(0, int(0.05 * len(tail)) - 1)],
            "p50_mhz": statistics.median(tail),
            "min_mhz": min(tail), "max_mhz": max(tail),
        }
        print(f"GPU {args.gpu} steady-state floor (p5, last 5min): "
              f"{result['steady_state']['p5_mhz']} MHz")
        print("Choose F_LOCK ~50 MHz BELOW the lowest p5 across sampled GPUs.")
    else:
        result["steady_state"] = None
        print("WARNING: no usable clock samples. Check SMI access before "
              "trusting anything else in this task.", file=sys.stderr)

    write_artifact(args.out, "01-floor", result)
    print(f"wrote {args.out}")


def cmd_lock(args):
    gpus = range(8) if args.all_gpus else [args.gpu]
    ok = True
    for g in gpus:
        print(f"locking GPU {g} -> {args.freq_mhz} MHz")
        ok &= set_perf_determinism(args.freq_mhz, None if args.all_gpus else g)
        if args.all_gpus:
            break  # global form applies to all
    print("locked" if ok else "LOCK FAILED — do not proceed to measurement")
    sys.exit(0 if ok else 1)


def cmd_verify(args):
    """An unloaded GPU reports the requested clock whether or not the lock
    is doing anything. Verification is only meaningful under load."""
    if not args.under_load:
        print("WARNING: verifying without load proves very little. "
              "Re-run with --under-load.", file=sys.stderr)

    import threading
    if args.under_load:
        threading.Thread(target=_sustained_load, args=(args.gpu, 30),
                         daemon=True).start()
        time.sleep(5)

    samples = [read_clocks(args.gpu) for _ in range(10)
               if not time.sleep(1)]
    observed = [s["sclk_mhz"] for s in samples if s.get("sclk_mhz")]
    if not observed:
        print("FAIL: no clock readings", file=sys.stderr)
        sys.exit(1)

    med = statistics.median(observed)
    drift = abs(med - args.freq_mhz)
    print(f"expected {args.freq_mhz} MHz, observed median {med} MHz "
          f"(drift {drift})")
    ok = drift <= args.tolerance_mhz
    print("PASS" if ok else f"FAIL: drift exceeds {args.tolerance_mhz} MHz")
    sys.exit(0 if ok else 1)


def cmd_stability(args):
    """Timing reproducibility across SEPARATE PROCESSES.

    Separate processes matter: in-process repetition hides allocator-state and
    context-setup variance that a real evaluation run will experience."""
    times = []
    for i in range(args.trials):
        out = subprocess.run(
            [sys.executable, __file__, "_probe", "--gpu", str(args.gpu)],
            capture_output=True, text=True)
        if out.returncode != 0:
            print(f"trial {i} failed: {out.stderr.strip()}", file=sys.stderr)
            continue
        times.append(float(out.stdout.strip()))

    if len(times) < 2:
        print("FAIL: insufficient successful trials", file=sys.stderr)
        sys.exit(1)

    mean = statistics.mean(times)
    cv = statistics.stdev(times) / mean
    result = {"gpu": args.gpu, "trials": len(times), "times_ms": times,
              "mean_ms": mean, "cv": cv, "gate": STABILITY_CV_GATE,
              "passed": cv < STABILITY_CV_GATE}
    write_artifact(args.out, "01-stability", result)
    print(f"CV = {cv:.4f} (gate {STABILITY_CV_GATE})")
    print("PASS" if result["passed"] else
          "FAIL — timing noise will swamp real differences. Investigate "
          "before proceeding: lock not holding? thermal? another process?")
    sys.exit(0 if result["passed"] else 1)


def cmd_interference(args):
    """Does load on sibling GPUs perturb timing? Shapes the whole schedule."""
    import threading

    lo, hi = (int(x) for x in args.load_gpus.split("-"))
    load_gpus = list(range(lo, hi + 1))

    print(f"baseline: timing GPU {args.timing_gpu}, siblings idle")
    quiet = [_timed_reference(args.timing_gpu) for _ in range(args.trials)]

    print(f"loaded: siblings {load_gpus} under sustained load")
    threads = [threading.Thread(target=_sustained_load, args=(g, 120),
                               daemon=True) for g in load_gpus]
    for t in threads:
        t.start()
    time.sleep(15)  # let siblings reach steady state
    busy = [_timed_reference(args.timing_gpu) for _ in range(args.trials)]

    q, b = statistics.median(quiet), statistics.median(busy)
    delta = (b - q) / q

    if abs(delta) < 0.01:
        verdict, consequence = "negligible", \
            "Sweeps and authoritative timing can share the node."
    elif abs(delta) < 0.03:
        verdict, consequence = "moderate", \
            "Authoritative runs (task 06 final pass) need a quiet node."
    else:
        verdict, consequence = "significant", \
            "Every timing run needs an idle node. Final timings serialize; " \
            "re-plan the schedule and record it in STATE.md."

    result = {"timing_gpu": args.timing_gpu, "load_gpus": load_gpus,
              "quiet_median_ms": q, "busy_median_ms": b,
              "delta_fraction": delta, "verdict": verdict,
              "scheduling_consequence": consequence,
              "quiet_ms": quiet, "busy_ms": busy}
    write_artifact(args.out, "01-interference", result)
    print(f"\nquiet {q:.4f} ms -> busy {b:.4f} ms  ({delta:+.2%})")
    print(f"verdict: {verdict}\n{consequence}")


def cmd_probe(args):
    print(_timed_reference(args.gpu))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("floor"); f.set_defaults(fn=cmd_floor)
    f.add_argument("--gpu", type=int, default=0)
    f.add_argument("--minutes", type=int, default=15)
    f.add_argument("--out", default="artifacts/01/floor.json")

    l = sub.add_parser("lock"); l.set_defaults(fn=cmd_lock)
    l.add_argument("--freq-mhz", type=int, required=True)
    l.add_argument("--gpu", type=int, default=0)
    l.add_argument("--all-gpus", action="store_true")

    v = sub.add_parser("verify"); v.set_defaults(fn=cmd_verify)
    v.add_argument("--freq-mhz", type=int, required=True)
    v.add_argument("--gpu", type=int, default=0)
    v.add_argument("--under-load", action="store_true")
    v.add_argument("--tolerance-mhz", type=int, default=50)

    s = sub.add_parser("stability"); s.set_defaults(fn=cmd_stability)
    s.add_argument("--gpu", type=int, default=0)
    s.add_argument("--trials", type=int, default=30)
    s.add_argument("--out", default="artifacts/01/stability.json")

    i = sub.add_parser("interference"); i.set_defaults(fn=cmd_interference)
    i.add_argument("--timing-gpu", type=int, default=0)
    i.add_argument("--load-gpus", default="1-7")
    i.add_argument("--trials", type=int, default=15)
    i.add_argument("--out", default="artifacts/01/interference.json")

    pr = sub.add_parser("_probe"); pr.set_defaults(fn=cmd_probe)
    pr.add_argument("--gpu", type=int, default=0)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
