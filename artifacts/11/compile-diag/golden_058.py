#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""float64 CPU golden for one failing workload of L2__058_mamba2_selective_scan.

Decides the question the diff alone cannot: is torch.compile LESS accurate than
eager, or MORE? Inputs are generated exactly as the harness generates them
(same seed, same prepare_inputs), then promoted to float64 and run on CPU
through the problem's own unmodified reference, so `dtype` inside the reference
is float64 and no bf16 rounding happens anywhere. Both GPU outputs are then
measured against that.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")

import torch  # noqa: E402
from _common import load_problem, prepare_inputs, exec_reference  # noqa: E402


def _compiled(definition):
    ns: dict = {}
    exec(compile(definition.reference, "<reference>", "exec"), ns)
    return torch.compile(ns["run"], dynamic=False), ns


def stats(got, gold, atol, rtol):
    g = got.detach().cpu().to(torch.float64)
    r = gold.detach().cpu().to(torch.float64)
    d = (g - r).abs()
    bad = d > (atol + rtol * r.abs())
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "rms": float((d * d).mean().sqrt()),
        "max_rel_vs_clamped_ref": float((d / r.abs().clamp(min=atol)).max()),
        "matched_ratio": 1.0 - float(bad.double().mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuid", required=True)
    ap.add_argument("--json-out", required=True)
    a = ap.parse_args()

    prob = Path("/work/data/SOL-ExecBench/benchmark/L2/058_mamba2_selective_scan")
    definition, workloads = load_problem(prob)
    w = [x for x in workloads if x.uuid == a.uuid][0]
    tol = w.tolerance
    atol, rtol = float(tol.max_atol), float(tol.max_rtol)

    ref_run, ns = exec_reference(definition)

    # The reference hard-codes `.float()` (= float32) for its SSM internals, so
    # promoting only the inputs raises `expected scalar type Double but found
    # Float` at the Y_off einsum. That is why artifacts/golden has no entry for
    # this problem. The golden therefore runs a source variant with `.float()`
    # -> `.to(torch.float64)`: the same mathematical function, evaluated at
    # higher working precision, which is exactly what a golden must be.
    src64 = definition.reference.replace(".float()", ".to(torch.float64)")
    assert src64 != definition.reference
    ns64: dict = {}
    exec(compile(src64, "<reference-f64>", "exec"), ns64)
    gold_run = ns64["run"]
    torch.manual_seed(0)
    ins = prepare_inputs(definition, w, ns, device="cuda:0")

    out_eager = ref_run(*ins).detach().clone()
    torch.cuda.synchronize()

    cmp_run, _ = _compiled(definition)
    out_comp = cmp_run(*ins).detach().clone()
    torch.cuda.synchronize()

    # float64 CPU golden from the SAME input bits.
    ins64 = tuple(
        (x.detach().cpu().to(torch.float64) if isinstance(x, torch.Tensor) else x)
        for x in ins
    )
    print("running float64 CPU golden ...", flush=True)
    gold = gold_run(*ins64)
    print("golden done", gold.dtype, gold.shape, flush=True)

    doc = {
        "uuid": a.uuid, "axes": dict(w.axes), "torch": torch.__version__,
        "tolerance": {"max_atol": atol, "max_rtol": rtol,
                      "required_matched_ratio": float(tol.required_matched_ratio)},
        "golden": {"dtype": str(gold.dtype), "device": "cpu",
                   "absmax": float(gold.abs().max())},
        "eager_vs_golden": stats(out_eager, gold, atol, rtol),
        "compiled_vs_golden": stats(out_comp, gold, atol, rtol),
        "compiled_vs_eager": stats(out_comp, out_eager, atol, rtol),
    }
    # who is closer, element by element
    ge = (out_eager.detach().cpu().to(torch.float64) - gold.to(torch.float64)).abs()
    gc = (out_comp.detach().cpu().to(torch.float64) - gold.to(torch.float64)).abs()
    doc["elementwise"] = {
        "frac_compiled_strictly_closer": float((gc < ge).double().mean()),
        "frac_eager_strictly_closer": float((ge < gc).double().mean()),
        "frac_tied": float((ge == gc).double().mean()),
    }

    # How far off is the tolerance? Multiplier on atol (rtol scaled with it)
    # needed to reach the required 0.99 matched_ratio, compiled vs eager.
    g = out_comp.detach().cpu().to(torch.float64)
    r = out_eager.detach().cpu().to(torch.float64)
    d = (g - r).abs()
    sweep = {}
    for k in (1, 2, 3, 4, 6, 8, 16):
        bad = d > (atol * k + rtol * k * r.abs())
        sweep[k] = 1.0 - float(bad.double().mean())
    doc["atol_multiplier_sweep_compiled_vs_eager"] = sweep
    gg = (g - gold.to(torch.float64)).abs()
    rr = (r - gold.to(torch.float64)).abs()
    doc["atol_multiplier_sweep_vs_golden"] = {
        k: {"eager": 1.0 - float((rr > (atol*k + rtol*k*gold.abs())).double().mean()),
            "compiled": 1.0 - float((gg > (atol*k + rtol*k*gold.abs())).double().mean())}
        for k in (1, 2, 3, 4, 6, 8, 16)}
    print(json.dumps(doc, indent=2))
    Path(a.json_out).write_text(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
