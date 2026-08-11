#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: why does torch.compile fail a workload's correctness check?

NOT a measurement runner. Produces no timing and no artifact that anything
scores against -- it answers one question: for a given problem/workload, how
far apart are the eager reference and the compiled reference, and how does that
compare to (a) the workload's tolerance and (b) eager's own run-to-run spread.

    python scripts/diag_compile_divergence.py \
        --problem data/L2/009_decoder_layer_with_residual_connections \
        --mode default --limit 4

Columns reported per workload:
  tol_atol/tol_rtol   the workload's calibrated tolerance
  eager_vs_eager      max abs error, same source run twice on fresh inputs
  compiled_vs_eager   max abs error, torch.compile output vs eager output
  matched_ratio       fraction of elements inside |d| <= atol + rtol*|ref|
  verdict             PASS/FAIL under the harness's own rule
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "runners"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _common import (  # noqa: E402
    exec_reference,
    load_problem,
    prepare_inputs,
    problem_key,
)

MODES = {"eager": None, "default": "__default__",
         "max-autotune": "max-autotune-no-cudagraphs"}


def _compiled_run(ref_src: str, mode: str | None):
    """Build `run` from source, then wrap it exactly as variants.py does."""
    import torch

    ns: dict = {}
    exec(compile(ref_src, "<reference>", "exec"), ns)
    fn = ns["run"]
    if mode is None:
        return fn, ns
    kw = {} if mode == "__default__" else {"mode": mode}
    return torch.compile(fn, dynamic=False, **kw), ns


def _as_tuple(out):
    if isinstance(out, (list, tuple)):
        return tuple(out)
    if isinstance(out, dict):
        return tuple(out[k] for k in sorted(out))
    return (out,)


def _stats(got, ref, atol: float, rtol: float):
    """The harness's own comparison, per output tensor, worst over outputs."""
    import torch

    from sol_execbench.core.bench.correctness import compute_error_stats
    from sol_execbench.core.data.workload import ToleranceSpec

    tol = ToleranceSpec(max_atol=atol, max_rtol=rtol, required_matched_ratio=0.99)
    worst = {"max_abs": 0.0, "max_rel": 0.0, "matched_ratio": 1.0, "exceeds": False}
    for g, r in zip(_as_tuple(got), _as_tuple(ref)):
        if not isinstance(g, torch.Tensor):
            continue
        c, exceeds = compute_error_stats(g.detach(), r.detach(), tol)
        # recompute matched_ratio (compute_error_stats does not return it)
        x, y = g.detach().to(torch.float32), r.detach().to(torch.float32)
        ae = (x - y).abs()
        bad = (ae > (atol + rtol * y.abs())) | ~torch.isfinite(ae)
        mr = 1.0 - float(bad.sum().item()) / float(ae.numel())
        worst["max_abs"] = max(worst["max_abs"], float(c.max_absolute_error or 0.0))
        worst["max_rel"] = max(worst["max_rel"], float(c.max_relative_error or 0.0))
        worst["matched_ratio"] = min(worst["matched_ratio"], mr)
        worst["exceeds"] = worst["exceeds"] or bool(exceeds)
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--mode", default="default", choices=list(MODES))
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--uuid", action="append", default=None)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    import torch

    prob = Path(a.problem).resolve()
    definition, workloads = load_problem(prob)
    if a.uuid:
        workloads = [w for w in workloads if w.uuid in set(a.uuid)]
    workloads = workloads[: a.limit]

    ref_run, ref_ns = exec_reference(definition)
    cmp_run, cmp_ns = _compiled_run(definition.reference, MODES[a.mode])

    rows = []
    for w in workloads:
        tolerance = w.tolerance
        atol = float(getattr(tolerance, "max_atol", 0.0) or 0.0)
        rtol = float(getattr(tolerance, "max_rtol", 0.0) or 0.0)

        torch.manual_seed(0)
        ins_a = prepare_inputs(definition, w, ref_ns, device=a.device)
        out_ref = ref_run(*ins_a)
        torch.cuda.synchronize()

        # eager against itself, fresh inputs from the same seed: the spread the
        # tolerance was calibrated on.
        torch.manual_seed(0)
        ins_b = prepare_inputs(definition, w, ref_ns, device=a.device)
        out_ref2 = ref_run(*ins_b)
        torch.cuda.synchronize()
        e2e = _stats(out_ref2, out_ref, atol, rtol)

        torch.manual_seed(0)
        ins_c = prepare_inputs(definition, w, cmp_ns, device=a.device)
        out_cmp = cmp_run(*ins_c)
        torch.cuda.synchronize()
        c2e = _stats(out_cmp, out_ref, atol, rtol)

        dtypes = sorted({str(v.dtype) for v in _as_tuple(out_ref)
                         if isinstance(v, torch.Tensor)})
        rows.append({
            "uuid": w.uuid, "axes": dict(w.axes), "out_dtypes": dtypes,
            "tol_atol": atol, "tol_rtol": rtol,
            "eager_vs_eager": e2e, "compiled_vs_eager": c2e,
            "verdict": "FAIL" if c2e["exceeds"] else "PASS",
        })
        r = rows[-1]
        print(f"{w.uuid[:8]} {str(dict(w.axes))[:44]:44s} "
              f"atol={atol:.3e} rtol={rtol:.3e} | "
              f"e/e={e2e['max_abs']:.3e} | c/e={c2e['max_abs']:.3e} "
              f"rel={c2e['max_rel']:.3e} mr={c2e['matched_ratio']:.6f} "
              f"-> {r['verdict']}", flush=True)

    doc = {"problem": problem_key(prob), "mode": a.mode,
           "torch": torch.__version__, "rows": rows}
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
