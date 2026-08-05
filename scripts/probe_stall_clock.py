#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""D20: does the 3.37x stall coincide with a GPU clock transition?

    env/solb python scripts/probe_stall_clock.py --gpu 1 --calls 40

`artifacts/02/timing-stall-probe.json` narrowed it hard. At mm[4096], 16 stall
iterations all landed between **3.35x and 3.40x** -- a 0.05-wide window. Random
contention, preemption or a noisy neighbour give a *spread* of magnitudes; a
fixed multiplier means the same kernel running a specific, reproducible slower
path. The rate is constant per iteration (0.135% at rep=100, 0.125% at rep=25),
the index is scattered, and all four GPUs show it equally.

A discrete clock step would explain a fixed multiplier exactly, and this part
has exactly one intermediate step to fall to:

    $ cat /sys/class/drm/card1/device/pp_dpm_sclk
    S: 38Mhz *
    1: 500Mhz
    2: 2200Mhz          (valid determinism range 500-1600)

The arithmetic does not land cleanly -- 1600/500 = 3.20 against an observed
3.37 -- so this is a hypothesis with a known discrepancy, not a conclusion.
Hence measuring instead of asserting: sample the active DPM level and the
instantaneous clock while the timing loop runs, and report whether any
excursion is observed at all.

Reading sysfs cannot prove causation: the sampler and the stall are not on a
common clock, so a coincidence in time is suggestive, not decisive. What it CAN
do is falsify -- if the clock never leaves its level across thousands of
iterations containing many stalls, a clock transition is not the mechanism and
the next suspect is kernel selection inside hipBLASLt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

LEVEL_RE = re.compile(r"^(\S+):\s*(\d+)Mhz\s*(\*?)", re.M)


def sample_clocks(card: Path, stop: threading.Event, out: list, period: float) -> None:
    """Poll the active DPM level and reported sclk until told to stop."""
    dpm = card / "device" / "pp_dpm_sclk"
    while not stop.is_set():
        try:
            txt = dpm.read_text()
            active = [(lvl, int(mhz)) for lvl, mhz, star in LEVEL_RE.findall(txt) if star]
            out.append((time.monotonic(), active[0] if active else None))
        except Exception:
            pass
        time.sleep(period)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1, help="torch/HIP index")
    ap.add_argument("--card", type=int, default=None,
                    help="override the DRM card; normally resolved by PCI bus")
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--period", type=float, default=0.001)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "02" / "timing-stall-clock.json")
    a = ap.parse_args()
    if a.gpu == 0:
        raise SystemExit("GPU 0 is reserved for authoritative timing; use 1-7.")
    # Resolve by PCI bus, never by index. The DRM numbering is unrelated to
    # every other ordering on this node -- torch 1 is card9, and card1 is
    # torch 0 -- so sampling `card{gpu}` would read a GPU that is idle while
    # the load runs elsewhere. That failure is silent and entirely plausible:
    # a flat clock trace, and a confident wrong conclusion that the clock is
    # not the mechanism. Same trap as D11 and the task-01 "floor" that was
    # fiction.
    if a.card is not None:
        card = Path(f"/sys/class/drm/card{a.card}")
    else:
        probe = subprocess.run(
            [str(ROOT / "env" / "solb"), "python", "-c",
             "import sys,json; sys.path.insert(0,'scripts');"
             "from gpu_map import torch_to_drm_card; print(json.dumps(torch_to_drm_card()))"],
            capture_output=True, text=True, timeout=600)
        if probe.returncode != 0:
            raise SystemExit(f"could not resolve DRM card:\n{probe.stderr[-2000:]}")
        mapping = json.loads(probe.stdout.strip().splitlines()[-1])
        if str(a.gpu) not in mapping:
            raise SystemExit(f"torch index {a.gpu} not in DRM map {mapping}")
        card = Path(mapping[str(a.gpu)])
    print(f"  torch gpu {a.gpu} -> {card}  (resolved by PCI bus)")
    if not (card / "device" / "pp_dpm_sclk").exists():
        raise SystemExit(f"{card}: no pp_dpm_sclk; pass --card explicitly")

    samples: list = []
    stop = threading.Event()
    t = threading.Thread(target=sample_clocks, args=(card, stop, samples, a.period),
                         daemon=True)
    t.start()

    code = f'''
import json, statistics, torch
from sol_execbench.core.bench.timing import time_runnable
a = torch.randn({a.size}, {a.size}, device="cuda")
b = torch.randn({a.size}, {a.size}, device="cuda")
import time
out = []
for call in range({a.calls}):
    t0 = time.monotonic()
    t = time_runnable(lambda a, b: torch.mm(a, b), [a, b], [], "cuda:0",
                      return_mode="all")
    t1 = time.monotonic()
    med = statistics.median(t)
    idx = [i for i, x in enumerate(t) if x > 2.0 * med]
    # CLOCK_MONOTONIC is system-wide on Linux, so these stamps are directly
    # comparable with the parent's clock sampler in the other process.
    out.append({{"call": call, "median_ms": med, "t0": t0, "t1": t1,
                 "times_ms": t,
                 "stalls": [{{"i": i, "ratio": round(t[i] / med, 3)}} for i in idx]}})
print("@@@" + json.dumps(out))
'''
    t0 = time.monotonic()
    proc = subprocess.run(
        [str(ROOT / "env" / "solb"), "python", "-c", code],
        capture_output=True, text=True, timeout=5400,
        env={**os.environ, "HIP_VISIBLE_DEVICES": str(a.gpu)})
    t1 = time.monotonic()
    stop.set(); t.join(timeout=2)

    if proc.returncode != 0:
        raise SystemExit(f"timing run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    line = [l for l in proc.stdout.splitlines() if l.startswith("@@@")]
    calls = json.loads(line[-1][3:])

    during = [s for s in samples if t0 <= s[0] <= t1 and s[1] is not None]
    levels = Counter(lvl for _, (lvl, _) in during)
    stalls = [st for c in calls for st in c["stalls"]]
    ratios = [st["ratio"] for st in stalls]

    # Align each timed call to the clock samples taken inside it. This is the
    # whole point of sharing CLOCK_MONOTONIC across the two processes: without
    # it, a wide clock histogram proves nothing, because process startup and
    # the idle gaps between calls are in the same window as the measurements.
    per_call = []
    for c in calls:
        mhz = [m for ts, (_, m) in during if c["t0"] <= ts <= c["t1"]]
        if not mhz:
            continue
        per_call.append({
            "call": c["call"], "n_clock": len(mhz),
            "clock_min": min(mhz), "clock_max": max(mhz),
            "clock_median": statistics.median(mhz),
            "clock_ratio": round(max(mhz) / min(mhz), 3) if min(mhz) else None,
            "median_ms": c["median_ms"],
            "n_stalls": len(c["stalls"]),
            "worst_ratio": max((s["ratio"] for s in c["stalls"]), default=None),
        })
    with_s = [p for p in per_call if p["n_stalls"]]
    without = [p for p in per_call if not p["n_stalls"]]

    print(f"  timing window      {t1 - t0:.1f}s, {len(calls)} calls")
    print(f"  clock samples      {len(during)} in-window "
          f"(~{len(during) / max(t1 - t0, 1e-9):.0f}/s)")
    print(f"  DPM level counts   {dict(levels)}")
    print(f"  stalls observed    {len(stalls)}"
          + (f"  ratios {min(ratios):.2f}-{max(ratios):.2f}x" if ratios else ""))
    print(f"\n  --- clock measured INSIDE each timed call ---")
    for label, grp in (("calls WITH a stall", with_s), ("calls without", without)):
        if not grp:
            print(f"  {label:22s} none")
            continue
        print(f"  {label:22s} n={len(grp):3d}  "
              f"clock {statistics.median([p['clock_min'] for p in grp]):.0f}-"
              f"{statistics.median([p['clock_max'] for p in grp]):.0f} MHz  "
              f"median in-call spread "
              f"{statistics.median([p['clock_ratio'] for p in grp]):.2f}x")

    if with_s and without:
        sr = statistics.median([p["clock_ratio"] for p in with_s])
        nr = statistics.median([p["clock_ratio"] for p in without])
        verdict = (
            f"calls containing a stall show a {sr:.2f}x in-call clock spread vs "
            f"{nr:.2f}x for calls without: the 'stall' is the clock, not the kernel"
            if sr > nr * 1.3 else
            f"in-call clock spread is comparable ({sr:.2f}x vs {nr:.2f}x); the "
            f"clock does not separate stalling calls from clean ones")
    elif not stalls:
        verdict = "no stalls captured in this window -- inconclusive"
    else:
        verdict = "every call contained a stall; no clean calls to compare against"
    print(f"\n  -> {verdict}")

    from provenance import write_artifact
    write_artifact(a.out, "02-timing-stall-clock", {
        "_note": "D20: sampling the DPM level during the timing loop to test "
                 "whether the fixed 3.37x stall is a clock transition. Falsifies "
                 "or fails to falsify; it cannot prove causation, because the "
                 "sampler and the GPU timer share no clock domain.",
        "gpu": a.gpu, "card": str(card), "size": a.size,
        "sample_period_s": a.period,
        "window_s": round(t1 - t0, 3),
        "clock_samples_in_window": len(during),
        "dpm_levels_seen": {str(k): v for k, v in levels.items()},
        "n_stalls": len(stalls),
        "stall_ratios": ratios,
        "stall_ratio_median": round(statistics.median(ratios), 3) if ratios else None,
        "verdict": verdict,
        "per_call_clock": per_call,
        "clock_series": [{"t": round(ts - t0, 4), "lvl": lvl, "mhz": m}
                         for ts, (lvl, m) in during],
        "calls": [{k: v for k, v in c.items() if k != "times_ms"} for c in calls],
    })
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
