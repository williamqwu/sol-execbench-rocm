#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing for the per-problem sweep runners.

Every runner honours the same contract (`scripts/runners/README.md`):

    --problem <dir> --out <file.json>

and **on failure still writes an output file recording the error**. A missing
file means "not yet run" and `shard_sweep.py` will redo it; a recorded failure
is a result. Prime directive 1: never let a failure disappear.

`run_guarded` implements exactly that: whatever the body raises, an artifact
lands on disk with the traceback in it.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from provenance import stamp  # noqa: E402


def problem_key(problem: Path) -> str:
    """'L1__rms_norm' — category and problem, the sweep's unit of coverage."""
    return f"{problem.parent.name}__{problem.name}"


def write_result(out: Path, kind: str, payload: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    body = {**stamp(kind), **payload}
    # Write-then-rename: a runner killed mid-write must not leave a truncated
    # JSON file that `already_done` would delete and redo forever, nor a
    # half-file that a later reader mistakes for a result.
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(body, f, indent=1, default=str)
    os.replace(tmp, out)


def run_guarded(out: Path, kind: str, body: Callable[[], dict[str, Any]]) -> int:
    """Run *body*, always writing an artifact. Returns a process exit code."""
    try:
        payload = body()
        payload.setdefault("ok", True)
    except BaseException as e:                     # noqa: BLE001 - deliberate
        payload = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
    write_result(out, kind, payload)
    return 0 if payload.get("ok") else 1


def workloads_path(problem: Path) -> Path:
    """Which workload.jsonl to run against.

    `SOLEXBENCH_WORKLOADS_ROOT` points at a tree of the same
    `<Category>/<problem>/workload.jsonl` shape -- in practice
    `artifacts/05/workloads/`, the dataset's workloads with AMD-DERIVED
    tolerances substituted in. Everything downstream of task 05 must run
    against those: the shipped file carries B200 tolerances, and scoring a
    kernel correct-or-not by a tolerance measured on other silicon is prime
    directive 2 in its most consequential form.

    Opt-in by environment rather than defaulted, so that a run against the
    shipped tolerances is a thing someone chose, and so the chosen root can be
    recorded in the artifact.
    """
    root = os.environ.get("SOLEXBENCH_WORKLOADS_ROOT")
    if root:
        cand = Path(root) / problem.parent.name / problem.name / "workload.jsonl"
        if cand.exists():
            return cand
        # Loud: a missing override file would otherwise silently fall back to
        # B200 tolerances for exactly the problems whose calibration is most
        # interesting.
        raise FileNotFoundError(
            f"SOLEXBENCH_WORKLOADS_ROOT={root} has no entry for "
            f"{problem.parent.name}/{problem.name} ({cand})")
    return problem / "workload.jsonl"


def load_problem(problem: Path):
    """(Definition, [Workload]) for a problem directory."""
    from sol_execbench.core import Definition, Workload

    definition = Definition(**json.loads((problem / "definition.json").read_text()))
    workloads = [
        Workload(**json.loads(line))
        for line in workloads_path(problem).read_text().splitlines()
        if line.strip()
    ]
    return definition, workloads


def reference_solution(definition, name_suffix: str = "reference", source: str | None = None):
    """A Solution that runs *source* (default: the problem's own reference).

    Used by task 02 (does the reference pass against itself on ROCm?), by task
    05 (reference-vs-reference variance) and by task 06 (T_b candidates, which
    pass their own source).
    """
    from sol_execbench.core import Solution

    return Solution(
        **{
            "name": f"{definition.name}__{name_suffix}",
            "definition": definition.name,
            "author": "sol-execbench-amd",
            "spec": {
                "languages": ["pytorch"],
                # LOCAL, not a named part: the runner must work on whichever
                # CDNA4 part the node has. Naming MI350X here would make the
                # artifact silently unusable on the MI355X node and vice versa.
                "target_hardware": ["LOCAL"],
                "entry_point": "kernel.py::run",
                "dependencies": ["torch"],
                "destination_passing_style": False,
            },
            "sources": [
                {"path": "kernel.py", "content": source or definition.reference}
            ],
        }
    )


def evaluate(definition, workloads, solution, config=None, timeout: int = 900,
             keep_staging: bool = False) -> list[dict]:
    """Evaluate *solution* and return traces as plain dicts.

    Runs through the real ProblemPackager + eval_driver subprocess, i.e. the
    same path the CLI takes, so anything measured here is measured under the
    harness's own isolation, cache-flush and anti-reward-hack machinery.
    """
    import subprocess

    from sol_execbench.core import BenchmarkConfig
    from sol_execbench.driver import ProblemPackager

    staging = Path(tempfile.mkdtemp(prefix="solb_run_", dir=os.environ.get(
        "SOLEXBENCH_SCRATCH", tempfile.gettempdir())))
    packager = ProblemPackager(
        definition=definition,
        workloads=workloads,
        solution=solution,
        config=config or BenchmarkConfig(),
        output_dir=staging,
        keep_output_dir=keep_staging,
    )
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    # The FlashInfer-Bench safetensors inputs are declared relative to the
    # dataset root, and the driver resolves them against the STAGING dir, not
    # the CWD. Without this, 9 of the 26 FlashInfer problems fail as ordinary
    # runtime errors and quietly leave the benchmark (STATE.md D5).
    env.setdefault("FLASHINFER_TRACE_DIR", str(ROOT))
    # Where the eval driver finds the rocprofiler shim when the rocprof
    # methodology is selected. Passed by path rather than assumed importable:
    # the driver runs from a staging directory with its own sys.path.
    env.setdefault("SOLEXBENCH_ROCPROF_SHIM", str(ROOT / "src" / "solexbench_rocm" / "shim"))

    if packager._is_cpp:
        cmd, _ = packager.compile()
        proc = subprocess.run(cmd, cwd=staging, capture_output=True, text=True,
                              timeout=timeout, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"compile failed:\n{proc.stderr[-4000:]}")

    proc = subprocess.run(packager.execute(), cwd=staging, capture_output=True,
                          text=True, timeout=timeout, env=env)
    traces = [json.loads(ln) for ln in proc.stdout.splitlines()
              if ln.strip().startswith("{")]
    if not traces:
        raise RuntimeError(
            f"no traces produced (rc={proc.returncode})\n"
            f"stderr:\n{proc.stderr[-4000:]}\nstdout:\n{proc.stdout[-2000:]}")
    return traces


def exec_reference(definition):
    """Exec a problem's reference and return (run_fn, namespace).

    The namespace matters: `custom_inputs_entrypoint` names a function defined
    alongside `run` in the same source, and 11,680 of the dataset's inputs are
    `custom` — generating them means calling that function.
    """
    ns: dict = {}
    exec(compile(definition.reference, f"<{definition.name}:reference>", "exec"), ns)
    run = ns.get("run")
    if run is None:
        raise RuntimeError("reference defines no top-level 'run'")
    return run, ns


def prepare_inputs(definition, workload, namespace, device: str = "cuda:0"):
    """Generate one workload's inputs exactly as the eval driver does.

    Mirrors the driver's three input paths rather than calling `gen_inputs`
    bare, because bare `gen_inputs` silently handles only the `random` and
    `scalar` cases. Across the dataset that would drop every `custom` input
    (11,680 of them) and every `safetensors` input (714, all FlashInfer-Bench)
    — i.e. it would fail on most of the benchmark while looking like an
    ordinary runtime error.
    """
    from sol_execbench.core.bench.io import gen_inputs, load_safetensors

    safe_tensors = {}
    if any(v.type == "safetensors" for v in workload.inputs.values()):
        # Same root priority as the driver: staging first, then the trace dir.
        roots = [ROOT, Path(os.environ.get("FLASHINFER_TRACE_DIR", ROOT))]
        safe_tensors = load_safetensors(definition, workload, roots)

    custom_fn = namespace.get(definition.custom_inputs_entrypoint) \
        if definition.custom_inputs_entrypoint else None

    return gen_inputs(
        definition,
        workload,
        device=device,
        safe_tensors=safe_tensors or None,
        custom_inputs_fn=custom_fn,
    )


# EvaluationStatus serializes as "PASSED", not "passed". Comparing against the
# lowercase spelling silently scores every passing workload as a failure, which
# looks exactly like a broken port -- so normalize once, here, and never
# compare status strings inline.
PASSED = "PASSED"


def _status(ev: dict) -> str | None:
    s = ev.get("status")
    return s.upper() if isinstance(s, str) else s


def _passed(ev: dict) -> bool:
    return _status(ev) == PASSED


def summarize(traces: list[dict]) -> dict:
    """Pass/fail counts and per-workload detail, from raw trace dicts."""
    per_workload = []
    for i, t in enumerate(traces):
        ev = t.get("evaluation") or {}
        perf = ev.get("performance") or {}
        corr = ev.get("correctness") or {}
        per_workload.append({
            "index": i,
            "workload_uuid": (t.get("workload") or {}).get("uuid"),
            "status": _status(ev),
            "latency_ms": perf.get("latency_ms"),
            "reference_latency_ms": perf.get("reference_latency_ms"),
            "max_absolute_error": corr.get("max_absolute_error"),
            "max_relative_error": corr.get("max_relative_error"),
            "has_nan": corr.get("has_nan"),
            "has_inf": corr.get("has_inf"),
            # Which timing methodology produced this number. Recorded on every
            # trace so hip_events and rocprof results can never be silently
            # mixed (tasks 02 and 04).
            "methodology": (ev.get("environment") or {}).get("methodology"),
            "log": (ev.get("log") or "")[-2000:] if not _passed(ev) else "",
        })
    passed = sum(1 for w in per_workload if w["status"] == PASSED)
    return {
        "workloads": len(per_workload),
        "passed": passed,
        "all_passed": passed == len(per_workload) and per_workload != [],
        "per_workload": per_workload,
    }
