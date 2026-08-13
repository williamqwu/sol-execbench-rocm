#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""D3 — is the clock lock costing us performance on THIS node?

The lock is `rocm-smi --setperfdeterminism 1600`, which on this part achieves
1300 MHz under a dense bf16 matrix-core load (task 01) and more under lighter
arithmetic (STATE.md D35). The claim this answers is a report from an MI355X
node that locking *degraded* performance. Whether that happens is a property of
the chassis and its power envelope, not of CDNA4, so it has to be measured
here.

Design, and why each part of it is there:

* **One card, interleaved in time.** Conditions alternate in ABBA order across
  blocks so that a drift in ambient temperature over the run cannot be read as
  a difference between conditions. Comparing two cards at once would confound
  the answer with the 1242-1307 MHz spread between cards that task 01 measured.
* **The rep is the unit of variance.** Locking buys stability, so a comparison
  that reports only means cannot answer the question it is asked.
* **The lock is restored on every exit path.** A run that dies having left the
  node unlocked silently invalidates whatever is measured next on it, so the
  restore is in a `finally` and in an `atexit`, and the final perf level is
  read back and printed.

    python scripts/clock_ab.py --gpu 7 --blocks 3 \
        --problems L1__074_fused_gated_mlp_silu,L2__036_convnextv2_layer_with_nhwc_persistence_backward \
        --out artifacts/12/clock-ab
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMI = "/opt/rocm/bin/rocm-smi"

# The node's standing policy, restored no matter how this script exits.
BASELINE_MHZ = 1600

CONDITIONS = {
    # name          -> (argv after `rocm-smi`, human description)
    "locked1600": (["--setperfdeterminism", "1600"],
                   "the node's standing policy: perf determinism capped at 1600"),
    "unlocked": (["--resetperfdeterminism"],
                 "governor back to auto -- boost as power and thermals allow"),
    "locked2200": (["--setperfdeterminism", "2200"],
                   "determinism mode at the part's max clock: separates "
                   "'determinism costs' from 'the 1600 cap costs'"),
}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def smi(args: list[str], gpu: int) -> str:
    proc = subprocess.run(["sudo", "-n", SMI, *args, "-d", str(gpu)],
                          capture_output=True, text=True)
    return proc.stdout.strip()


def perf_level(gpu: int) -> str:
    out = subprocess.run([SMI, "--showperflevel", "-d", str(gpu)],
                         capture_output=True, text=True).stdout
    # Only the GPU[n] rows -- rocm-smi's own banner contains the words
    # "Performance Level" too, and matching it records the banner as the level.
    for line in out.splitlines():
        if line.startswith("GPU[") and "Performance Level" in line:
            return line.split(":")[-1].strip()
    return "<unreadable>"


def apply_condition(name: str, gpu: int, settle_s: float) -> str:
    args, _ = CONDITIONS[name]
    smi(args, gpu)
    time.sleep(settle_s)
    level = perf_level(gpu)
    say(f"  condition {name}: perf level now {level!r}")
    return level


def restore(gpu: int) -> None:
    smi(["--setperfdeterminism", str(BASELINE_MHZ)], gpu)
    time.sleep(2)
    say(f"RESTORED gpu {gpu}: perf level {perf_level(gpu)!r} "
        f"(setperfdeterminism {BASELINE_MHZ})")


def probe(problems: list[str], gpu: int, out: Path, reps: int, hold: float,
          warmup: float, workload: int, timeout: int) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    inner = (f"HIP_VISIBLE_DEVICES={gpu} python /work/scripts/bounds/clock_ab_probe.py "
             f"--problem {','.join(problems)} --reps {reps} --hold {hold} "
             f"--warmup-s {warmup} --workload {workload} --out {out}")
    try:
        proc = subprocess.run([str(ROOT / "env" / "solb"), "bash", "-c", inner],
                              capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, "PATH": os.environ.get("PATH", "")})
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout}s"}
    if not out.exists():
        return {p: {"ok": False, "error": f"no artifact (rc={proc.returncode})",
                    "stderr_tail": proc.stderr[-2000:]} for p in problems}
    return json.loads(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--problems", required=True,
                    help="comma-separated problem keys, or a path to a JSON list")
    ap.add_argument("--conditions", default="locked1600,unlocked")
    ap.add_argument("--blocks", type=int, default=3,
                    help="how many times to visit every condition")
    ap.add_argument("--reps", type=int, default=5, help="reps per probe")
    ap.add_argument("--hold", type=float, default=4.0)
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--workload", type=int, default=-1)
    ap.add_argument("--settle-s", type=float, default=8.0,
                    help="wait after changing the clock policy")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    if a.problems.endswith(".json"):
        problems = json.loads(Path(a.problems).read_text())
    else:
        problems = [p for p in a.problems.split(",") if p]
    conditions = [c for c in a.conditions.split(",") if c]
    for c in conditions:
        if c not in CONDITIONS:
            print(f"unknown condition {c!r}; known: {sorted(CONDITIONS)}",
                  file=sys.stderr)
            return 2

    atexit.register(restore, a.gpu)
    say(f"gpu {a.gpu}: starting perf level {perf_level(a.gpu)!r}")

    results: list[dict] = []
    try:
        for block in range(a.blocks):
            # ABBA: reverse the visiting order on odd blocks so a monotone
            # drift over the run cancels instead of loading onto one condition.
            order = conditions if block % 2 == 0 else list(reversed(conditions))
            for cond in order:
                say(f"block {block}, condition {cond}")
                level = apply_condition(cond, a.gpu, a.settle_s)
                out = a.out / cond / f"block{block}.json"
                t0 = time.time()
                payloads = probe(problems, a.gpu, out, a.reps, a.hold,
                                 a.warmup_s, a.workload, a.timeout)
                for problem in problems:
                    payload = payloads.get(problem, {"ok": False,
                                                     "error": "missing from artifact"})
                    row = {
                        "block": block, "condition": cond, "problem": problem,
                        "perf_level": level, "gpu": a.gpu,
                        "ok": payload.get("ok"),
                        "error": payload.get("error"),
                        "ms_per_call_p50": payload.get("ms_per_call_p50"),
                        "ms_per_call_cv": payload.get("ms_per_call_cv"),
                        "gfx_mhz_p50": payload.get("gfx_mhz_p50_of_reps"),
                        "power_w_mean": payload.get("power_w_mean_of_reps"),
                        "input_dtypes": payload.get("input_dtypes"),
                        "workload_index": payload.get("workload_index"),
                    }
                    results.append(row)
                    say("    " + json.dumps({k: row[k] for k in
                        ("problem", "ms_per_call_p50", "gfx_mhz_p50",
                         "power_w_mean", "ok")}))
                say(f"  block {block} / {cond} took {time.time() - t0:.0f}s")
                (a.out / "rows.json").write_text(json.dumps(results, indent=1))
    finally:
        restore(a.gpu)

    (a.out / "rows.json").write_text(json.dumps(results, indent=1))
    say(f"wrote {a.out / 'rows.json'} ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
