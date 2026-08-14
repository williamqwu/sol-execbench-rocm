#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The feedback channel an agent gets: evaluate its current solution on hardware.

Invoked as ``./verify`` from inside a task packet. Compiles and runs the
solution against every workload, then prints a report that says specifically
what went wrong -- which workload, which error against which tolerance, the
compiler's message if it did not build.

Three things this is deliberately **not**:

- *Not the score.* ``scripts/score_solutions.py`` re-evaluates the harvested
  solution from a pristine tree on the authoritative GPU. That number counts;
  this one is for the agent to learn from. They can differ legitimately: this
  runs on a shared-pool GPU whose clock is the same but whose neighbours are
  busy.
- *Not unlimited.* Attempts are counted in ``.attempts.json`` and refused past
  the budget. An agent with unbounded hardware access converges by brute force,
  which measures the harness's patience rather than the model.
- *Not silent about its own failures.* Every attempt is appended to the log
  whether it passed, failed, or crashed the harness, because "how many times did
  it try" is part of the result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Resolved from this file's own location, so the same script works whether it is
# run from the repo or from the reduced verify root a task packet points at (see
# task_packet.build_verify_root). The reduced tree holds the evaluation code and
# nothing else -- no tolerance derivations, no analytic bounds, no other problem.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "runners"))
sys.path.insert(0, str(ROOT / "src"))
# _common resolves FlashInfer safetensors against this; in the reduced tree it is
# a symlink to the real blobs.
os.environ.setdefault("FLASHINFER_TRACE_DIR", str(ROOT))


def load_solution(packet: Path):
    """Load ``solution.json``, resolving empty ``content`` from ``path``.

    Reimplemented rather than imported from ``sol_execbench.cli.main``, whose
    ``_load_solution`` is private and belongs to the vendored upstream fork --
    depending on it here would couple the agent harness to an internal name that
    tracks upstream. The behaviour is small and worth owning: a source entry
    whose ``content`` is empty is filled from the file at ``path``, so an agent
    can write ``kernel.py`` normally instead of embedding escaped JSON.
    """
    from sol_execbench.core import Solution

    sol_path = packet / "solution.json"
    if not sol_path.exists():
        raise FileNotFoundError(
            "no solution.json in this directory yet -- write one first "
            "(see TASK.md for the schema)"
        )
    raw = json.loads(sol_path.read_text())
    for src in raw.get("sources", []):
        if not src.get("content"):
            f = packet / src["path"]
            if not f.exists():
                raise FileNotFoundError(
                    f"solution.json lists source {src['path']!r} with empty "
                    f"content and no such file exists in the packet"
                )
            src["content"] = f.read_text()
    return Solution(**raw)


def attempts_state(packet: Path) -> dict:
    p = packet / ".attempts.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {"used": 0, "log": []}


def save_attempts(packet: Path, state: dict) -> None:
    (packet / ".attempts.json").write_text(json.dumps(state, indent=2, default=str))


def _fmt(value, spec: str = ".3f", dash: str = "-") -> str:
    if value is None:
        return dash
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def report(summary: dict, workloads, elapsed: float, remaining: int) -> str:
    lines = []
    lines.append("")
    lines.append(f"  {'#':>3}  {'status':<22} {'yours(ms)':>10} {'ref(ms)':>10} "
                 f"{'speedup':>8}  {'max_abs':>10} {'allow_atol':>10} "
                 f"{'max_rel':>10} {'allow_rtol':>10}")
    lines.append("  " + "-" * 104)
    for w in summary["per_workload"]:
        i = w["index"]
        tol = {}
        if i < len(workloads):
            tol = (workloads[i].get("tolerance") or {})
        lat, ref = w.get("latency_ms"), w.get("reference_latency_ms")
        speedup = f"{ref / lat:.2f}x" if (lat and ref) else "-"
        lines.append(
            f"  {i:>3}  {str(w['status']):<22} {_fmt(lat):>10} {_fmt(ref):>10} "
            f"{speedup:>8}  {_fmt(w.get('max_absolute_error'), '.3e'):>10} "
            f"{_fmt(tol.get('max_atol'), '.3e'):>10} "
            f"{_fmt(w.get('max_relative_error'), '.3e'):>10} "
            f"{_fmt(tol.get('max_rtol'), '.3e'):>10}"
        )

    lines.append("")
    lines.append(f"  {summary['passed']}/{summary['workloads']} workloads passed "
                 f"in {elapsed:.1f}s")

    failed = [w for w in summary["per_workload"] if w["status"] != "PASSED"]
    if failed:
        lines.append("")
        lines.append(f"  {len(failed)} failing workload(s); logs for the first 3:")
        for w in failed[:3]:
            lines.append("")
            lines.append(f"  --- workload {w['index']} ({w['status']}) ---")
            log = (w.get("log") or "").strip()
            lines.append("\n".join(f"    {ln}" for ln in log.splitlines()[-40:])
                         or "    (no log captured)")

    lines.append("")
    lines.append(f"  attempts remaining after this one: {remaining}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", required=True, type=Path)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    packet: Path = args.packet.resolve()
    manifest = json.loads((packet / ".packet.json").read_text())
    max_attempts = int(manifest["max_attempts"])

    state = attempts_state(packet)
    if state["used"] >= max_attempts:
        print(
            f"\n  No verification attempts left ({state['used']}/{max_attempts} used).\n"
            f"  Whatever solution.json holds now is what gets scored, so make it\n"
            f"  your best reasoned answer -- review the reference once more rather\n"
            f"  than guessing at another variant.\n",
            file=sys.stderr,
        )
        return 2

    state["used"] += 1
    attempt_no = state["used"]
    remaining = max_attempts - attempt_no
    print(f"  verify attempt {attempt_no}/{max_attempts} "
          f"(GPU {os.environ.get('HIP_VISIBLE_DEVICES', '?')})")

    from _common import evaluate, summarize  # noqa: E402

    from sol_execbench.core import BenchmarkConfig, Definition, Workload  # noqa: E402

    # benchmark_reference defaults to False, which leaves reference_latency_ms at
    # 0.0 and gives the agent no idea whether its kernel is faster than the thing
    # it is replacing. TASK.md promises "your latency beside the reference's", so
    # it has to be asked for.
    config = BenchmarkConfig(benchmark_reference=True)

    definition = Definition(**json.loads((packet / "definition.json").read_text()))
    raw_workloads = [
        json.loads(ln)
        for ln in (packet / "workload.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    workloads = [Workload(**w) for w in raw_workloads]

    entry = {"attempt": attempt_no, "utc": time.time()}
    started = time.time()
    try:
        solution = load_solution(packet)
        entry["languages"] = [x.value for x in solution.spec.languages]
        traces = evaluate(definition, workloads, solution, config=config,
                          timeout=args.timeout)
        summary = summarize(traces)
        elapsed = time.time() - started
        entry.update(
            ok=True,
            passed=summary["passed"],
            workloads=summary["workloads"],
            all_passed=summary["all_passed"],
            elapsed_s=elapsed,
        )
        print(report(summary, raw_workloads, elapsed, remaining))
        rc = 0 if summary["all_passed"] else 1
    except Exception as exc:  # noqa: BLE001 - every failure is a result
        elapsed = time.time() - started
        entry.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                     elapsed_s=elapsed)
        # The message is the whole point of this path: a compile failure or a
        # schema error is the most actionable feedback an agent can get, and
        # swallowing it would leave it guessing.
        print(f"\n  Verification could not complete after {elapsed:.1f}s:\n",
              file=sys.stderr)
        print(f"    {type(exc).__name__}: {exc}\n", file=sys.stderr)
        print(f"  attempts remaining after this one: {remaining}\n", file=sys.stderr)
        rc = 1
    finally:
        state["log"].append(entry)
        save_attempts(packet, state)

    return rc


if __name__ == "__main__":
    sys.exit(main())
