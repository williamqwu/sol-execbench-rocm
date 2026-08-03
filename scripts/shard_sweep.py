#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shard a per-problem sweep across GPUs. Resumable by construction.

    python scripts/shard_sweep.py --task references \
        --category L1 L2 Quant FlashInfer-Bench --gpus 1-7 \
        --out artifacts/02/references/

Default is all four categories (all 235 problems). Narrowing --category is a
scope reduction; do it only when a task explicitly says to.

Resumability is not a nicety here. These sweeps run for hours, sessions get
interrupted, and prime directive 7 says never restart with different settings.
A problem whose output file already exists is skipped, so re-invoking the exact
same command continues where it stopped.

    --dry-run   print the work plan without touching a GPU (works on CPU)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

TASK_RUNNERS = {
    "references":          "runners/run_reference.py",
    "tolerances":          "runners/calibrate_tolerance.py",
    "tb-candidates":       "runners/time_tb_candidates.py",
    "methodology-compare": "runners/compare_methodology.py",
}


def parse_gpus(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-"))
            out.extend(range(lo, hi + 1))
        elif part:
            out.append(int(part))
    return out


def discover_problems(data_dir: Path, categories: list[str]) -> list[Path]:
    problems: list[Path] = []
    for cat in categories:
        cat_dir = data_dir / cat
        if not cat_dir.is_dir():
            print(f"WARNING: category dir missing: {cat_dir}", file=sys.stderr)
            continue
        for p in sorted(cat_dir.iterdir()):
            if (p / "definition.json").exists():
                problems.append(p)
    return problems


def already_done(out_dir: Path, problem: Path) -> bool:
    f = out_dir / f"{problem.parent.name}__{problem.name}.json"
    if not f.exists():
        return False
    try:
        json.loads(f.read_text())
        return True
    except Exception:
        f.unlink()          # truncated by an interrupted run; redo it
        return False


def run_one(problem: Path, gpu: int, runner: Path, out_dir: Path,
            extra: list[str], timeout: int) -> tuple[Path, bool, str]:
    out_file = out_dir / f"{problem.parent.name}__{problem.name}.json"
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu))
    cmd = [sys.executable, str(runner), "--problem", str(problem),
           "--out", str(out_file), *extra]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        out_file.write_text(json.dumps(
            {"problem": problem.name, "correctness_passed": False,
             "error": f"timeout after {timeout}s", "gpu": gpu}, indent=2))
        return problem, False, "timeout"
    if r.returncode != 0 and not out_file.exists():
        # Record the failure rather than losing it -- prime directive 1.
        out_file.write_text(json.dumps(
            {"problem": problem.name, "correctness_passed": False,
             "error": r.stderr.strip()[-4000:], "gpu": gpu}, indent=2))
        return problem, False, "error"
    return problem, True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASK_RUNNERS))
    ap.add_argument("--category", nargs="+",
                    default=["L1", "L2", "Quant", "FlashInfer-Bench"])
    ap.add_argument("--gpus", default="1-7")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs="*",
                    help="passed through to the runner (e.g. --seeds 10)")
    a = ap.parse_args()

    gpus = parse_gpus(a.gpus)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    runner = ROOT / "scripts" / TASK_RUNNERS[a.task]

    problems = discover_problems(Path(a.data), a.category)
    pending = [p for p in problems if not already_done(out_dir, p)]
    done = len(problems) - len(pending)

    print(f"task        {a.task}")
    print(f"runner      {runner}")
    print(f"gpus        {gpus}")
    print(f"problems    {len(problems)} total, {done} already done, "
          f"{len(pending)} pending")
    if pending:
        print(f"per-gpu     ~{len(pending) // max(1, len(gpus))} problems")

    if a.dry_run:
        print("\n-- dry run, nothing executed --")
        for i, p in enumerate(pending[:10]):
            print(f"  gpu {gpus[i % len(gpus)]}  {p.parent.name}/{p.name}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
        return
    if not runner.exists():
        sys.exit(f"runner not found: {runner}\n"
                 f"Implement it as part of the task that needs it.")
    if not pending:
        print("nothing to do")
        return

    start = time.time()
    ok = failed = 0
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(run_one, p, gpus[i % len(gpus)], runner, out_dir,
                        a.extra, a.timeout)
            for i, p in enumerate(pending)
        ]
        for n, fut in enumerate(futures, 1):
            problem, success, status = fut.result()
            ok, failed = ok + success, failed + (not success)
            elapsed = time.time() - start
            eta = elapsed / n * (len(pending) - n)
            print(f"[{n}/{len(pending)}] {status:<8} {problem.name}  "
                  f"(ok {ok} / fail {failed}, eta {eta/60:.0f}m)", flush=True)

    print(f"\ndone: {ok} ok, {failed} failed, {(time.time()-start)/60:.1f} min")
    if failed:
        print("Failures are recorded as artifacts, not lost. Triage them and "
              "record in STATE.md before marking the task done.")


if __name__ == "__main__":
    main()
