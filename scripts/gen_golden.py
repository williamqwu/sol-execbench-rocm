#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate float64 CPU golden references. NO GPU REQUIRED.

Why these exist: run-to-run comparison on AMD tells you a kernel is *stable*,
not that it is *correct*. A deterministically wrong kernel looks perfectly
stable. Golden references separate two very different findings during task 05:

    AMD differs from NVIDIA  -> expected, benign, a numerics difference
    AMD differs from correct -> a bug

Runs each problem's reference in float64 on CPU, per workload, and stores the
outputs keyed by workload uuid — the same key task 05 compares against.

    python scripts/gen_golden.py --out artifacts/golden --jobs 64

Cost control, stated rather than hidden: float64 on CPU is 30-100x slower than
bf16 on the GPU, and the dataset contains GEMMs at n=28672. Workloads above
``--max-gflop`` are SKIPPED and the skip is recorded per workload, so a missing
golden is always visible as a decision. Silently dropping the big ones would
leave exactly the compute-heavy problems -- the ones most likely to accumulate
error -- unchecked.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "runners"))
sys.path.insert(0, str(ROOT / "src"))


def _elements(definition, workload) -> int:
    """Total input+output elements — a cheap proxy for cost."""
    total = 0
    try:
        shapes = definition.get_input_shapes(workload.axes)
        for shape in shapes.values():
            n = 1
            for d in shape:
                n *= int(d)
            total += n
    except Exception:
        pass
    return total


def gen_one(args) -> tuple[str, dict]:
    problem, out_dir, max_elements = args
    key = f"{problem.parent.name}__{problem.name}"
    out_file = out_dir / f"{key}.pt"
    report: dict = {"problem": key, "workloads": {}}
    if out_file.exists():
        report["status"] = "cached"
        return key, report

    try:
        import torch

        from _common import exec_reference, load_problem, prepare_inputs

        torch.set_num_threads(1)      # one thread per worker; the pool is the
                                      # parallelism, and oversubscription here
                                      # makes the whole sweep slower.
        definition, workloads = load_problem(problem)
        run, ns = exec_reference(definition)

        goldens: dict = {}
        for wl in workloads:
            n = _elements(definition, wl)
            if max_elements and n > max_elements:
                report["workloads"][wl.uuid] = f"skipped: {n} elements > cap"
                continue
            # Two tiers, and WHICH ONE RAN IS RECORDED, because they are not
            # equally strong evidence:
            #
            #   float64    -- arithmetic ground truth. A disagreement is a bug.
            #   native_cpu -- same dtypes, CPU kernels. Still an independent
            #                 implementation with a different accumulation
            #                 order, so it catches a deterministically wrong
            #                 GPU kernel; but a disagreement here can also be
            #                 ordinary low-precision noise.
            #
            # The fallback exists because many references construct internal
            # tensors at a literal dtype (`torch.zeros(..., bfloat16)`, weights
            # made inside `run`), so promoting only the inputs mixes dtypes and
            # raises. Silently skipping those would drop the fused, multi-step
            # problems -- exactly the ones where error accumulates.
            for mode in ("float64", "native_cpu"):
                try:
                    torch.manual_seed(0)
                    inputs = prepare_inputs(definition, wl, ns, device="cpu")
                    if mode == "float64":
                        # Integer and boolean tensors are left alone:
                        # promoting an index tensor changes the program.
                        inputs = [
                            t.to(torch.float64)
                            if isinstance(t, torch.Tensor) and t.is_floating_point()
                            else t
                            for t in inputs
                        ]
                    with torch.no_grad():
                        out = run(*inputs)
                    if isinstance(out, torch.Tensor):
                        out = [out]
                    elif isinstance(out, dict):
                        out = list(out.values())
                    goldens[wl.uuid] = {
                        "mode": mode,
                        "outputs": [
                            t.detach().cpu()
                            for t in out
                            if isinstance(t, torch.Tensor)
                        ],
                    }
                    report["workloads"][wl.uuid] = f"ok:{mode}"
                    break
                except Exception as e:               # noqa: BLE001
                    report["workloads"][wl.uuid] = f"{type(e).__name__}: {e}"

        if goldens:
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(goldens, out_file)
        report["status"] = "ok" if goldens else "empty"
        report["n_ok"] = sum(
            1 for v in report["workloads"].values() if str(v).startswith("ok")
        )
        report["n_workloads"] = len(workloads)
        return key, report
    except Exception as e:                           # noqa: BLE001
        report["status"] = "failed"
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()[-1500:]
        return key, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--out", default="artifacts/golden")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--max-elements", type=int, default=64_000_000,
                    help="skip workloads above this input-element count "
                         "(0 = no cap). Skips are recorded, never silent.")
    ap.add_argument("--category", nargs="+",
                    default=["L1", "L2", "Quant", "FlashInfer-Bench"])
    a = ap.parse_args()

    data, out = Path(a.data), Path(a.out)
    problems = [p for c in a.category for p in sorted((data / c).glob("*"))
                if (p / "definition.json").exists()]
    print(f"{len(problems)} problems, {a.jobs} workers, "
          f"element cap {a.max_elements or 'none'}")
    if not problems:
        sys.exit("no problems found — check --data")

    reports = {}
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for key, report in ex.map(
            gen_one, [(p, out, a.max_elements) for p in problems]
        ):
            reports[key] = report
            if report.get("status") in ("ok", "cached"):
                ok += 1
            else:
                fail += 1
                print(f"  FAIL {key}: {report.get('error', report.get('status'))}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "_report.json").write_text(json.dumps(reports, indent=1))
    print(f"\n{ok} problems with goldens, {fail} without -> {out}")
    print("Missing goldens are not fatal; task 05 falls back to run-to-run "
          "comparison for those and records `vs_golden: null`. The per-workload "
          f"reasons are in {out}/_report.json.")


if __name__ == "__main__":
    main()
