#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate float64 CPU golden references. NO GPU REQUIRED.

Why these exist: run-to-run comparison on AMD tells you a kernel is *stable*,
not that it is *correct*. A deterministically wrong kernel looks perfectly
stable. Golden references separate two very different findings during task 05:

    AMD differs from NVIDIA  -> expected, benign, a numerics difference
    AMD differs from correct -> a bug

Runs each problem's reference in float64 on CPU and stores the output. Slow and
embarrassingly parallel; run it on any machine, not the MI355X node.

    python scripts/gen_golden.py --data data/... --out artifacts/golden/ --jobs 16
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def gen_one(problem: Path, out_dir: Path) -> tuple[str, bool, str]:
    out_file = out_dir / f"{problem.parent.name}__{problem.name}.pt"
    if out_file.exists():
        return problem.name, True, "cached"
    try:
        import torch
        sys.path.insert(0, str(problem))
        spec = json.loads((problem / "definition.json").read_text())

        # Reference modules expose run(); inputs come from get_inputs() when the
        # problem needs structured data (paged KV, sparse masks), else from the
        # definition's tensor specs.
        import importlib.util
        ref_path = problem / "reference.py"
        m_spec = importlib.util.spec_from_file_location("ref", ref_path)
        mod = importlib.util.module_from_spec(m_spec)
        m_spec.loader.exec_module(mod)

        torch.manual_seed(0)
        inputs = mod.get_inputs() if hasattr(mod, "get_inputs") else None
        if inputs is None:
            raise NotImplementedError(
                "no get_inputs(); build inputs from definition tensor specs — "
                "see reference/upstream-audit.md for the schema")

        # float64 on CPU is the whole point: it is the arithmetic ground truth,
        # independent of any GPU's accumulation order.
        inputs = [t.double().cpu() if hasattr(t, "double") else t for t in inputs]
        with torch.no_grad():
            out = mod.run(*inputs)

        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"outputs": out, "definition": spec.get("name"),
                    "dtype": "float64", "device": "cpu", "seed": 0}, out_file)
        return problem.name, True, "ok"
    except Exception as e:
        return problem.name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--out", default="artifacts/golden")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--category", nargs="+",
                    default=["L1", "L2", "Quant", "FlashInfer-Bench"])
    a = ap.parse_args()

    data, out = Path(a.data), Path(a.out)
    problems = [p for c in a.category for p in sorted((data / c).glob("*"))
                if (p / "definition.json").exists()]
    print(f"{len(problems)} problems, {a.jobs} workers")
    if not problems:
        sys.exit("no problems found — check --data")

    ok = fail = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for name, success, msg in ex.map(gen_one, problems, [out] * len(problems)):
            ok, fail = ok + success, fail + (not success)
            if not success:
                print(f"  FAIL {name}: {msg.splitlines()[0]}")
    print(f"\n{ok} ok, {fail} failed -> {out}")
    if fail:
        print("Missing goldens are not fatal; task 05 falls back to run-to-run "
              "comparison for those problems. Record which ones.")


if __name__ == "__main__":
    main()
