#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Re-time a run across all eight cards at once, one problem per card.

    python scripts/retime_parallel.py --run artifacts/10/gpt56-180 \
        --part MI355X --gpus 0,1,2,3,4,5,6,7 [--only-missing]

**This is a departure from the discipline and it needs its evidence attached.**
`CLAUDE.md` §4 says GPU 0 is for authoritative timing and nothing else, and
what task 01 settled is narrower than "the node is fine": it measured **-0.11%**
for one authoritative timing running beside *sweep* work on the other cards.
Eight simultaneous timing runs is a different load -- every card drawing at once
is exactly the condition under which the node's power envelope could pull clocks
down, and `STATE.md` D35 is the entry about how much a clock error moves a
bound.

So it is not assumed. `artifacts/11/parallel-retime-validation.json` re-measures
control problems that already have a solo GPU-0 number, under full 8-wide load,
and compares. Run that first. If the controls move by more than run-to-run
noise, this script is the wrong tool and the serial path is the only one.

What stays true either way: **one job per card.** The parallelism here is across
cards, never within one. Each worker owns a card for the life of its problem,
and the per-measurement exclusivity check still runs -- a foreign container on
*this* card is still a contaminated measurement, whatever the other seven are
doing.

Writes `<run>/retimed/<key>.json`, the same artifacts `agent_score.py
--reuse-retimed` reads, so scoring afterwards is unchanged and single-threaded.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from tolerance_roots import (
    TOLERANCE_ROOTS,
    container_tolerance_root,
    recorded_tolerance_root,
)
from verify_artifacts import artifact_part

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("SOLEXBENCH_SCRATCH", "/var/tmp/solbench"))

_print_lock = threading.Lock()


def say(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def foreign_on(gpu: int) -> list[str]:
    """Foreign processes on the card HIP index *gpu* names, via the container."""
    try:
        proc = subprocess.run(
            [str(ROOT / "env" / "solb"), "python",
             "/work/scripts/gpu_exclusive.py", "--gpu", "0"],
            env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home()),
                 "HIP_VISIBLE_DEVICES": str(gpu)},
            capture_output=True, text=True, timeout=120)
    except Exception as exc:                                  # noqa: BLE001
        return [f"check failed: {type(exc).__name__}: {exc}"]
    if proc.returncode == 0:
        return []
    return [ln.strip() for ln in proc.stderr.splitlines() if ln.strip()]


def measure(key: str, kernel: str, out: Path, gpu: int, iterations: int,
            warmup: int, timeout: int, n_cards: int,
            tolerance_root: str) -> bool:
    cat, name = key.split("__", 1)
    staged = SCRATCH / "retime-par" / f"{key}.json"
    staged.parent.mkdir(parents=True, exist_ok=True)
    if staged.exists():
        staged.unlink()

    foreign = foreign_on(gpu)
    cmd = [str(ROOT / "env" / "solb"), "python", "/work/scripts/agent_eval.py",
           "--problem", f"/work/data/SOL-ExecBench/benchmark/{cat}/{name}",
           "--kernel", kernel, "--out", str(staged),
           "--iterations", str(iterations), "--warmup", str(warmup),
           "--timeout", str(max(60, timeout - 120)), "--quiet"]
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home()),
           "HIP_VISIBLE_DEVICES": str(gpu),
           "SOLEXBENCH_WORKLOADS_ROOT": tolerance_root}
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=timeout)
        rc, err = proc.returncode, proc.stderr
    except subprocess.TimeoutExpired:
        rc, err = -1, f"timed out after {timeout}s"

    if not staged.exists():
        out.write_text(json.dumps({
            "ok": False, "error": f"runner produced no artifact (rc={rc})",
            "stderr_tail": (err or "")[-3000:],
            "per_workload": [], "workloads": 0, "passed": 0,
            "retimed_gpu": gpu, "concurrent_cards": n_cards,
            "authoritative_card_exclusive": not foreign,
            "tolerance_root": tolerance_root}, indent=1))
        return False

    payload = json.loads(staged.read_text())
    payload["tolerance_root"] = tolerance_root
    payload["retimed_gpu"] = gpu
    # Recorded because it is a property of HOW this number was taken, and a
    # reader comparing it with a serially-measured one deserves to know without
    # having to find the commit.
    payload["concurrent_cards"] = n_cards
    payload["authoritative_card_exclusive"] = not foreign
    if foreign:
        payload["authoritative_card_shared_with"] = foreign
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    return bool(payload.get("passed"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=lambda p: Path(p).resolve())
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--part", required=True, choices=sorted(TOLERANCE_ROOTS),
                    help="GPU part being measured; selects that part's "
                         "correctness-tolerance tree")
    ap.add_argument("--only-missing", action="store_true",
                    help="skip problems that already have a retimed artifact")
    ap.add_argument("--only", action="append", default=[])
    a = ap.parse_args()
    tolerance_root = container_tolerance_root(a.part)

    run = json.loads((a.run / "run.json").read_text())
    gpus = [int(g) for g in a.gpus.split(",") if g.strip() != ""]
    retimed = a.run / "retimed"
    retimed.mkdir(parents=True, exist_ok=True)
    kernels = a.run / "kernels"
    kernels.mkdir(parents=True, exist_ok=True)

    work: list[tuple[str, str]] = []
    for key, sess in sorted(run["sessions"].items()):
        if a.only and key not in set(a.only):
            continue
        existing = retimed / f"{key}.json"
        if a.only_missing and existing.exists():
            try:
                prior = json.loads(existing.read_text())
                recorded_root = recorded_tolerance_root(
                    prior,
                    artifact_part(prior) if isinstance(prior, dict) else None,
                )
            except (OSError, json.JSONDecodeError):
                recorded_root = None
            if recorded_root == tolerance_root:
                continue
            say(f"RE-TIME {key}: existing artifact has tolerance_root="
                f"{recorded_root!r}, expected {tolerance_root!r}")
        # The path handed to the container must be one the container has.
        # SOLEXBENCH_SCRATCH is bind-mounted at the SAME absolute path inside
        # and out, which is why the sandbox path works verbatim; the repo is at
        # /work, so a host path under artifacts/ does not. Getting this wrong
        # fails as `FileNotFoundError` on the kernel, which reads like a missing
        # submission rather than a mount mistake.
        sandbox_kernel = Path(sess.get("sandbox", "")) / "kernel.py"
        saved = kernels / f"{key}.py"
        if sandbox_kernel.exists():
            saved.write_text(sandbox_kernel.read_text())
            container_path = str(sandbox_kernel)
        elif saved.exists():
            container_path = f"/work/{saved.relative_to(ROOT)}"
        else:
            say(f"SKIP {key}: no kernel in the sandbox and none saved")
            continue
        work.append((key, container_path))

    say(f"{len(work)} problems over {len(gpus)} cards")
    q: queue.Queue = queue.Queue()
    for item in work:
        q.put(item)

    counts = {"done": 0, "failed": 0}
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                key, kernel = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            ok = measure(key, kernel, retimed / f"{key}.json", gpu,
                         a.iterations, a.warmup, a.timeout, len(gpus),
                         tolerance_root)
            with lock:
                counts["done"] += 1
                if not ok:
                    counts["failed"] += 1
                n = counts["done"]
            say(f"gpu{gpu} {key[:52]:52} {time.time()-t0:6.0f}s "
                f"({n}/{len(work)})")

    threads = [threading.Thread(target=worker, args=(g,), daemon=True)
               for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    say(f"done: {counts['done']} measured, {counts['failed']} produced nothing")
    say("now score it, single-threaded: "
        f"python3 scripts/agent_score.py --run {a.run.relative_to(ROOT)} "
        f"--part {a.part} --reuse-retimed --manifest MANIFEST-FOR-{a.part}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
