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

_SMI = None
_SMI_TRIED = False


def _amdsmi():
    """Initialise amdsmi once. Repeated amdsmi_init() per sample is wasteful
    and, at 1 Hz over 15 minutes, needless churn in a library we are trusting
    for the most consequential measurement in the project."""
    global _SMI, _SMI_TRIED
    if not _SMI_TRIED:
        _SMI_TRIED = True
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            _SMI = amdsmi
        except Exception:
            _SMI = None
    return _SMI


def _temp_c(smi, handle):
    """MI355X does NOT support the EDGE sensor -- it raises NOT_SUPPORTED.
    HOTSPOT (junction) is the one that reads, and is what rocm-smi prints.

    This matters more than it looks: the original code read EDGE inside the
    same try block as the clock read, so an unsupported sensor discarded the
    SCLK sample too, and the floor measurement would have silently produced
    zero usable samples."""
    for sensor in ("HOTSPOT", "JUNCTION", "EDGE"):
        st = getattr(smi.AmdSmiTemperatureType, sensor, None)
        if st is None:
            continue
        try:
            return smi.amdsmi_get_temp_metric(
                handle, st, smi.AmdSmiTemperatureMetric.CURRENT), sensor
        except Exception:
            continue
    return None, None


def read_clocks(gpu: int) -> dict:
    """Return {sclk_mhz, mclk_mhz, power_w, temp_c} for torch device *gpu*.

    *gpu* is a torch device index. It is resolved to the amdsmi handle by PCI
    identity, NOT by position -- see scripts/gpu_map.py; the two orderings are
    scrambled relative to each other on this node.
    """
    smi = _amdsmi()
    if smi is not None:
        try:
            from gpu_map import amdsmi_handle
            handle = amdsmi_handle(gpu)
            sclk = smi.amdsmi_get_clock_info(handle, smi.AmdSmiClkType.GFX)
            mclk = smi.amdsmi_get_clock_info(handle, smi.AmdSmiClkType.MEM)
            power = smi.amdsmi_get_power_info(handle)
            temp, sensor = _temp_c(smi, handle)
            return {
                "sclk_mhz": sclk.get("clk"), "mclk_mhz": mclk.get("clk"),
                "sclk_locked": sclk.get("clk_locked"),
                "power_w": power.get("current_socket_power"),
                "temp_c": temp, "temp_sensor": sensor, "source": "amdsmi",
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


PERF_LEVEL_GLOB = "/sys/class/drm/card*/device/power_dpm_force_performance_level"


def perf_levels() -> dict[str, str]:
    """Current power_dpm_force_performance_level for every card."""
    import glob
    out = {}
    for f in sorted(glob.glob(PERF_LEVEL_GLOB)):
        try:
            out[f] = Path(f).read_text().strip()
        except Exception as e:
            out[f] = f"<unreadable: {e}>"
    return out


def set_perf_determinism(freq_mhz: int, gpu: int | None = None) -> bool:
    """AMD's documented determinism mechanism: cap the soft max clock.

    Verifies the effect rather than trusting the exit status. Observed on this
    node: inside a stock Docker container /sys is mounted read-only, and
    `rocm-smi --setperfdeterminism` then exits 0 having done NOTHING and
    printed no error. A silent no-op here is the worst possible outcome --
    every subsequent measurement would be taken at an unlocked boost clock
    while the artifacts claim F_LOCK.
    """
    before = perf_levels()
    cmd = ["rocm-smi", "--setperfdeterminism", str(freq_mhz)]
    if gpu is not None:
        cmd += ["-d", str(gpu)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(f"  FAILED: {out.stderr.strip()}", file=sys.stderr)
        return False

    after = perf_levels()
    locked = {f for f in after if after[f] == "perf_determinism"}
    if not locked:
        print("  FAILED: exit status 0 but no card reports "
              "'perf_determinism'. Levels are still: "
              f"{sorted(set(after.values()))}.\n"
              "  Most likely /sys is read-only (stock container) or this user "
              "lacks privileges. Do NOT proceed: an unverified lock means "
              "every timing is taken at an unknown clock.", file=sys.stderr)
        return False

    # A partial lock is the dangerous case: some GPUs held at F_LOCK, others
    # boosting freely, with nothing in the artifacts to distinguish them.
    if gpu is None and len(locked) != len(after):
        unlocked = sorted(f for f in after if f not in locked)
        print(f"  FAILED: asked to lock every GPU but only {len(locked)}/"
              f"{len(after)} report 'perf_determinism'. Still unlocked: "
              f"{unlocked}", file=sys.stderr)
        return False

    print(f"  {len(locked)}/{len(after)} card(s) at perf_determinism")
    return True


def reset_clocks() -> None:
    subprocess.run(["rocm-smi", "-r"], capture_output=True)


# --------------------------------------------------------------------------
# Load generation
# --------------------------------------------------------------------------

def _sustained_load(gpu: int, seconds: float, size: int = 8192,
                    stop: "object | None" = None):
    """Saturate the matrix cores with back-to-back BF16 GEMMs.

    *stop* is an optional threading.Event allowing the caller to end the loop
    early and then join. Without that, the interpreter can shut down while this
    thread is still inside a HIP call, which aborts the process
    ("terminate called without an active exception") *after* the artifact is
    written -- leaving a good result behind a non-zero exit status, which any
    sweep runner would score as a failure.
    """
    import torch
    dev = torch.device(f"cuda:{gpu}")
    a = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=dev, dtype=torch.bfloat16)
    deadline = time.time() + seconds
    while time.time() < deadline:
        if stop is not None and stop.is_set():
            break
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
    done = threading.Event()    # load thread has exited
    halt = threading.Event()    # ask the load thread to exit

    def load():
        try:
            _sustained_load(args.gpu, total, stop=halt)
        except Exception as e:
            print(f"load thread died: {e}", file=sys.stderr)
        finally:
            done.set()

    t = threading.Thread(target=load, daemon=True)
    t.start()

    siblings = [g for g in range(args.n_gpus) if g != args.gpu]

    # Optional worst-case condition. Floors measured with siblings idle are the
    # best case; tasks 05/06 shard across GPUs 1-7, so for much of the project
    # the node is fully loaded. If the two floors differ, F_LOCK must suit the
    # busy one -- raising F_LOCK later invalidates everything measured before.
    sib_procs = []
    if args.load_siblings:
        print(f"loading siblings {siblings} for the whole run")
        sib_procs = [_spawn_load(g, total + 60) for g in siblings]
        time.sleep(30)   # let the node reach power/thermal steady state
    samples = []
    start = time.time()
    while time.time() - start < total and not done.is_set():
        s = read_clocks(args.gpu)
        s["t"] = time.time() - start
        # This node is shared. If somebody else's job lands on a sibling GPU
        # mid-run, it may couple into our floor through the power budget --
        # and we would never know from the target GPU's samples alone.
        s["sibling_power_w"] = [read_clocks(g).get("power_w") for g in siblings]
        samples.append(s)
        time.sleep(1.0 / SAMPLE_HZ)

    # Wind the load down and join before touching the artifact: letting the
    # interpreter tear down under an in-flight HIP call aborts the process
    # after a successful write.
    halt.set()
    t.join(timeout=120)
    if t.is_alive():
        print("WARNING: load thread did not exit within 120 s", file=sys.stderr)
    for p in sib_procs:
        p.terminate()
    for p in sib_procs:
        try:
            p.wait(timeout=30)
        except Exception:
            p.kill()

    tail_start = max(0, total - 300)
    tail = [s["sclk_mhz"] for s in samples
            if s.get("t", 0) >= tail_start and s.get("sclk_mhz")]

    # Idle MI355X draws ~240 W; a busy one draws far more. Flag any sibling
    # that was clearly working during the tail window we derive the floor from.
    tail_sib = [s.get("sibling_power_w") or [] for s in samples
                if s.get("t", 0) >= max(0, total - 300)]
    busy_sibs = sorted({siblings[i] for row in tail_sib
                        for i, p in enumerate(row)
                        if p and p > 400 and i < len(siblings)})

    result = {"gpu": args.gpu, "minutes": args.minutes,
              "n_samples": len(samples), "n_tail": len(tail),
              "siblings_loaded_deliberately": bool(args.load_siblings),
              "siblings_busy_during_tail": busy_sibs,
              "samples": samples}
    if busy_sibs and not args.load_siblings:
        print(f"WARNING: sibling GPU(s) {busy_sibs} were under load during the "
              f"window this floor is derived from. Another user shares this "
              f"node; treat this floor as contaminated unless that load was "
              f"yours.", file=sys.stderr)
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
    halt = threading.Event()
    loader = None
    if args.under_load:
        loader = threading.Thread(
            target=_sustained_load, args=(args.gpu, 60),
            kwargs={"stop": halt}, daemon=True)
        loader.start()
        time.sleep(5)

    samples = [read_clocks(args.gpu) for _ in range(10)
               if not time.sleep(1)]

    # Join before exiting: tearing the interpreter down under an in-flight HIP
    # call aborts the process (exit 134), which would make a PASS read as a
    # failure to anything checking the exit status.
    halt.set()
    if loader is not None:
        loader.join(timeout=120)
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


def _spawn_load(gpu: int, seconds: float) -> subprocess.Popen:
    """Sibling load in a SEPARATE PROCESS.

    Threads cannot do this job: the GIL serialises the Python-level launch
    loop, so seven 'loaded' siblings would in practice be intermittently idle
    and the interference measured would understate reality. This experiment
    decides whether authoritative timing can share the node, so an
    understated answer is the dangerous direction to err in.
    """
    return subprocess.Popen(
        [sys.executable, __file__, "_load", "--gpu", str(gpu),
         "--seconds", str(seconds)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def cmd_interference(args):
    """Does load on sibling GPUs perturb timing? Shapes the whole schedule."""
    lo, hi = (int(x) for x in args.load_gpus.split("-"))
    load_gpus = list(range(lo, hi + 1))

    print(f"baseline: timing GPU {args.timing_gpu}, siblings idle")
    quiet = [_timed_reference(args.timing_gpu) for _ in range(args.trials)]

    print(f"loaded: siblings {load_gpus} under sustained load")
    load_seconds = 120 + 30 * args.trials
    procs = [_spawn_load(g, load_seconds) for g in load_gpus]
    time.sleep(30)  # let siblings reach thermal/power steady state

    sib = [read_clocks(g) for g in load_gpus]
    sib_power = [s.get("power_w") for s in sib if s.get("power_w")]
    print(f"  sibling power now: {sib_power} W")
    dead = [g for g, p in zip(load_gpus, procs) if p.poll() is not None]
    if dead:
        print(f"  WARNING: load process died on GPU(s) {dead} — the 'busy' "
              f"condition is not what it claims", file=sys.stderr)

    busy = [_timed_reference(args.timing_gpu) for _ in range(args.trials)]

    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=30)
        except Exception:
            p.kill()

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
              "sibling_power_w_under_load": sib_power,
              "load_processes_died": dead,
              "quiet_ms": quiet, "busy_ms": busy}
    write_artifact(args.out, "01-interference", result)
    print(f"\nquiet {q:.4f} ms -> busy {b:.4f} ms  ({delta:+.2%})")
    print(f"verdict: {verdict}\n{consequence}")


def cmd_probe(args):
    print(_timed_reference(args.gpu))


def cmd_load(args):
    """Sustained load worker, used as a subprocess by `interference`."""
    _sustained_load(args.gpu, args.seconds)


def cmd_determinism_sweep(args):
    """Map REQUESTED determinism clock -> ACHIEVED clock under sustained load.

    Exists because on MI350X the two are not the same number. Requesting
    1250 MHz produced 1049 MHz under load, even though the same GPU sustained
    1335-1390 MHz *unlocked* at the same 1000 W cap. So the lock is not simply
    a ceiling here: entering performance-determinism mode changes the
    voltage/frequency operating point, and under a power cap that can leave
    less frequency available, not more.

    This matters far beyond a failed check. F_LOCK is defined as a clock the
    hardware actually holds. If the requested number and the held number
    differ, then picking F_LOCK from the unlocked floor -- which is what
    tasks/01 step 2 says to do, and what worked on the 1400 W MI355X -- yields
    a setting the part cannot honour, and every subsequent measurement would
    be taken at an unknown clock while the artifacts claim otherwise.

    So the relationship is measured rather than assumed, and F_LOCK is chosen
    from the ACHIEVED column.
    """
    import threading

    rows = []
    for req in args.freqs:
        set_perf_determinism(req, None)
        halt = threading.Event()
        loader = threading.Thread(
            target=_sustained_load, args=(args.gpu, args.seconds + 60),
            kwargs={"stop": halt}, daemon=True)
        loader.start()
        time.sleep(args.settle)          # thermal and power steady state
        samples = [read_clocks(args.gpu) for _ in range(args.samples)
                   if not time.sleep(1)]
        halt.set()
        loader.join(timeout=120)

        clk = [s["sclk_mhz"] for s in samples if s.get("sclk_mhz")]
        pw = [s.get("power_w") for s in samples if s.get("power_w")]
        row = {
            "requested_mhz": req,
            "achieved_median_mhz": statistics.median(clk) if clk else None,
            "achieved_min_mhz": min(clk) if clk else None,
            "achieved_p5_mhz": sorted(clk)[max(0, len(clk) // 20)] if clk else None,
            "median_power_w": statistics.median(pw) if pw else None,
            "n_samples": len(clk),
        }
        rows.append(row)
        print(f"  requested {req:>5} -> median "
              f"{row['achieved_median_mhz']} MHz, min {row['achieved_min_mhz']}, "
              f"{row['median_power_w']} W", flush=True)

    write_artifact(args.out, "01-determinism-sweep",
                   {"gpu": args.gpu, "rows": rows,
                    "note": "requested vs achieved under sustained BF16 GEMM; "
                            "F_LOCK must be chosen from the achieved column"})
    print(f"wrote {args.out}")


def _achieved_at(gpu: int, setpoint: int, settle: int, samples: int) -> dict:
    """Apply *setpoint* to one GPU and measure what it holds under load.

    One GPU at a time and one setpoint at a time, on purpose. The original
    node-wide sweep stepped through eleven frequencies in a single run and
    recorded GPU 0 holding 1214 MHz at a 1500 setpoint; measured in isolation the
    same GPU holds 1495. The low figure was the part still settling toward the new
    operating point, not the part refusing the request — a sweep that does not
    settle between steps measures its own step order (D28).
    """
    import threading

    # `-d N` in rocm-smi is a ROCM-SMI index; *gpu* here is a TORCH index, and on
    # this node torch 0 is rocm-smi 3. Passing the torch index straight through
    # sets a different card than the one being loaded and sampled, so the
    # measurement reads whatever setpoint that card happened to carry. Observed:
    # GPU 0 "held" 1807 MHz at a 1480 setpoint, because 1480 went to rocm-smi 0
    # while torch 0 was still at a 2100 setpoint from an earlier sweep and pinned
    # to its power cap. This is the error scripts/gpu_map.py exists to prevent.
    from gpu_map import torch_to_rocm_smi

    set_perf_determinism(setpoint, torch_to_rocm_smi()[gpu])
    halt = threading.Event()
    loader = threading.Thread(target=_sustained_load, args=(gpu, settle + samples + 60),
                              kwargs={"stop": halt}, daemon=True)
    loader.start()
    time.sleep(settle)
    got = [read_clocks(gpu) for _ in range(samples) if not time.sleep(1)]
    halt.set()
    loader.join(timeout=120)

    clk = [s["sclk_mhz"] for s in got if s.get("sclk_mhz")]
    pw = [s.get("power_w") for s in got if s.get("power_w")]
    return {
        "setpoint_mhz": setpoint,
        "achieved_median_mhz": statistics.median(clk) if clk else None,
        "achieved_min_mhz": min(clk) if clk else None,
        "median_power_w": statistics.median(pw) if pw else None,
        "n_samples": len(clk),
    }


def cmd_equalize(args):
    """Find the per-GPU setpoint that makes every GPU hold the same clock.

    Why this exists. A single node-wide setpoint does not give a single clock on
    this node: at 1650 the eight GPUs hold 1318-1644 MHz, a 25% spread, and the
    six slow ones are not power-limited -- they draw 949-995 W of a 1400 W cap.
    Two obey the request and six land at about 0.82x it, at every setpoint tried.

    The consequence is not cosmetic. A T_b re-timed on GPU 5 is ~20% slower than
    the same code on GPU 0, so authoritative timing has to be pinned to one GPU,
    and one GPU means the T_b pass is serial: ~20 hours where eight GPUs would be
    two and a half, with seven cards idle throughout.

    The fix follows from the ratios being *stable*. If GPU 2 holds 0.82x its
    setpoint, then asking it for 1900 yields the 1480 that GPU 0 yields at 1500 --
    so a common achieved clock is reachable by giving each GPU its own setpoint.
    This calibrates that per GPU by secant search on the measured response, which
    is close to linear over the useful range (GPU 2: 1800 -> 1426, 1900 -> 1480,
    2000 -> 1550, 2100 -> 1600).

    What it costs, stated plainly: the common clock has to be one the *slowest*
    GPU can reach, so it is below what GPUs 0 and 1 could hold alone. That is the
    right trade anyway -- an F_LOCK that 8 of 8 cards hold is a better basis for a
    benchmark than one 2 of 8 hold, because reproducibility is the entire purpose
    of locking the clock. Absolute times get slower; SOL scores are within-platform
    ratios and do not care.
    """
    target = args.target_mhz
    tol = args.tolerance_mhz
    results: dict[int, dict] = {}

    print(f"equalizing {args.n_gpus} GPU(s) to {target} +/- {tol} MHz under load\n")
    for gpu in range(args.n_gpus):
        # Start from the identity guess and one scaled guess, then secant. Two
        # measurements bracket every GPU seen here: the obedient ones need ~target
        # and the rest ~target/0.82.
        probes: list[tuple[int, float]] = []
        for setpoint in (target, int(round(target / 0.82 / 10) * 10)):
            if any(p[0] == setpoint for p in probes):
                continue
            row = _achieved_at(gpu, setpoint, args.settle, args.samples)
            got = row["achieved_median_mhz"]
            print(f"  GPU {gpu}: set {setpoint:>5} -> {got} MHz "
                  f"({row['median_power_w']} W)")
            if got is None:
                continue
            probes.append((setpoint, got))
            if abs(got - target) <= tol:
                break

        best = min(probes, key=lambda p: abs(p[1] - target)) if probes else None
        for _ in range(args.max_iters):
            if best is None or abs(best[1] - target) <= tol:
                break
            if len(probes) >= 2:
                (x0, y0), (x1, y1) = probes[-2], probes[-1]
                slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.82
                if abs(slope) < 0.1:
                    slope = 0.82
            else:
                slope = 0.82
            nxt = int(round((best[0] + (target - best[1]) / slope) / 5) * 5)
            nxt = max(500, min(2400, nxt))
            if any(p[0] == nxt for p in probes):
                break
            row = _achieved_at(gpu, nxt, args.settle, args.samples)
            got = row["achieved_median_mhz"]
            print(f"  GPU {gpu}: set {nxt:>5} -> {got} MHz "
                  f"({row['median_power_w']} W)")
            if got is None:
                break
            probes.append((nxt, got))
            best = min(probes, key=lambda p: abs(p[1] - target))

        if best is None:
            print(f"  GPU {gpu}: FAILED to measure")
            results[gpu] = {"ok": False}
            continue
        row = _achieved_at(gpu, best[0], args.settle, args.samples)
        within = abs((row["achieved_median_mhz"] or 0) - target) <= tol
        results[gpu] = {
            "setpoint_mhz": best[0],
            "achieved_median_mhz": row["achieved_median_mhz"],
            "achieved_min_mhz": row["achieved_min_mhz"],
            "median_power_w": row["median_power_w"],
            "within_tolerance": within,
            "probes": [{"setpoint_mhz": s, "achieved_mhz": a} for s, a in probes],
            "ok": within,
        }
        print(f"  GPU {gpu}: SETTLED at setpoint {best[0]} -> "
              f"{row['achieved_median_mhz']} MHz "
              f"{'OK' if within else 'OUT OF TOLERANCE'}\n")

    ok = [g for g, r in results.items() if r.get("ok")]
    achieved = [r["achieved_median_mhz"] for r in results.values()
                if r.get("achieved_median_mhz")]
    spread = (max(achieved) - min(achieved)) if achieved else None
    print(f"{len(ok)}/{args.n_gpus} GPUs within {tol} MHz of {target}")
    if spread is not None:
        print(f"spread across GPUs: {spread:.0f} MHz "
              f"({100 * spread / target:.2f}% of target)")
    print("\nsetpoints:")
    for g in sorted(results):
        r = results[g]
        print(f"  GPU {g}: setpoint {r.get('setpoint_mhz')} -> "
              f"{r.get('achieved_median_mhz')} MHz")

    write_artifact(args.out, "01-equalize", {
        "target_mhz": target,
        "tolerance_mhz": tol,
        "n_gpus": args.n_gpus,
        "per_gpu": {str(g): r for g, r in results.items()},
        "all_within_tolerance": len(ok) == args.n_gpus,
        "spread_mhz": spread,
        "note": "per-GPU determinism setpoints chosen so every GPU HOLDS the same "
                "clock under load. A single node-wide setpoint does not: two GPUs "
                "obey it and six land at ~0.82x, and the six are not power-limited. "
                "An F_LOCK 8/8 cards hold is a better basis than one 2/8 hold.",
    })
    print(f"\nwrote {args.out}")
    sys.exit(0 if len(ok) == args.n_gpus else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    eq = sub.add_parser("equalize", help="per-GPU setpoints for one common clock")
    eq.add_argument("--target-mhz", type=int, required=True)
    eq.add_argument("--tolerance-mhz", type=int, default=15)
    eq.add_argument("--n-gpus", type=int, default=8)
    eq.add_argument("--settle", type=int, default=25)
    eq.add_argument("--samples", type=int, default=12)
    eq.add_argument("--max-iters", type=int, default=4)
    eq.add_argument("--out", default="artifacts/01/equalized-clocks.json")
    eq.set_defaults(fn=cmd_equalize)

    f = sub.add_parser("floor"); f.set_defaults(fn=cmd_floor)
    f.add_argument("--gpu", type=int, default=0)
    f.add_argument("--minutes", type=int, default=15)
    f.add_argument("--n-gpus", type=int, default=8,
                   help="node GPU count, for sibling-contention sampling")
    f.add_argument("--load-siblings", action="store_true",
                   help="also load every other GPU: the worst-case floor, "
                        "which is the condition tasks 05/06 actually run in")
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

    ds = sub.add_parser("determinism-sweep"); ds.set_defaults(fn=cmd_determinism_sweep)
    ds.add_argument("--gpu", type=int, default=0)
    ds.add_argument("--freqs", type=int, nargs="+", required=True)
    ds.add_argument("--settle", type=int, default=30)
    ds.add_argument("--samples", type=int, default=20)
    ds.add_argument("--seconds", type=int, default=60)
    ds.add_argument("--out", default="artifacts/01/determinism-sweep.json")

    pr = sub.add_parser("_probe"); pr.set_defaults(fn=cmd_probe)
    pr.add_argument("--gpu", type=int, default=0)

    ld = sub.add_parser("_load"); ld.set_defaults(fn=cmd_load)
    ld.add_argument("--gpu", type=int, default=0)
    ld.add_argument("--seconds", type=float, default=60)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
