#!/usr/bin/env python3
"""Adversarial-verifier variant of scripts/diag_compile_divergence.py.

Per output tensor: dtype, tolerance, eager-vs-eager, compiled-vs-eager,
matched_ratio, and the smallest k in {1,2,4,...,2^20} such that scaling BOTH
atol and rtol by k makes matched_ratio >= 0.99. One workload per process, run
first, so torch._dynamo never hits its recompile limit.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")
from _common import exec_reference, load_problem, prepare_inputs, problem_key  # noqa

MODES = {"eager": None, "default": "__default__",
         "max-autotune": "max-autotune-no-cudagraphs"}


def _as_tuple(out):
    if isinstance(out, (list, tuple)):
        return tuple(out)
    if isinstance(out, dict):
        return tuple(out[k] for k in sorted(out))
    return (out,)


def _names(definition, n):
    try:
        ks = list(definition.outputs.keys())
    except Exception:
        ks = []
    if len(ks) == n:
        return ks
    return [f"out{i}" for i in range(n)]


def mr(x, y, atol, rtol):
    import torch
    a, b = x.detach().to(torch.float64), y.detach().to(torch.float64)
    ae = (a - b).abs()
    bad = (ae > (atol + rtol * b.abs())) | ~torch.isfinite(ae)
    return 1.0 - float(bad.sum().item()) / float(ae.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--uuid", required=True)
    ap.add_argument("--mode", default="default", choices=list(MODES))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    import torch

    prob = Path(a.problem).resolve()
    definition, workloads = load_problem(prob)
    w = [x for x in workloads if x.uuid == a.uuid][0]
    atol = float(getattr(w.tolerance, "max_atol", 0.0) or 0.0)
    rtol = float(getattr(w.tolerance, "max_rtol", 0.0) or 0.0)
    cap = getattr(w.tolerance, "max_error_cap", None)

    ref_run, ref_ns = exec_reference(definition)
    ns2: dict = {}
    exec(compile(definition.reference, "<reference>", "exec"), ns2)
    fn = ns2["run"]
    m = MODES[a.mode]
    cmp_run = fn if m is None else (torch.compile(fn, dynamic=False) if m == "__default__"
                                    else torch.compile(fn, dynamic=False, mode=m))

    torch.manual_seed(0)
    ins = prepare_inputs(definition, w, ref_ns, device=a.device)
    ref = _as_tuple(ref_run(*ins)); torch.cuda.synchronize()
    ref = tuple(t.detach().clone() if isinstance(t, torch.Tensor) else t for t in ref)
    torch.manual_seed(0)
    ins2 = prepare_inputs(definition, w, ref_ns, device=a.device)
    ref2 = _as_tuple(ref_run(*ins2)); torch.cuda.synchronize()
    torch.manual_seed(0)
    ins3 = prepare_inputs(definition, w, ns2, device=a.device)
    got = _as_tuple(cmp_run(*ins3)); torch.cuda.synchronize()

    names = _names(definition, len(ref))
    rows = []
    for nm, g, r, r2 in zip(names, got, ref, ref2):
        if not isinstance(g, torch.Tensor):
            continue
        gf, rf = g.detach().to(torch.float64), r.detach().to(torch.float64)
        ae = (gf - rf).abs()
        finite = torch.isfinite(ae)
        row = {
            "output": nm, "dtype": str(r.dtype), "numel": int(r.numel()),
            "ref_absmax": float(rf.abs().max().item()),
            "eager_vs_eager_maxabs": float((r2.detach().to(torch.float64) - rf).abs().max().item()),
            "eager_vs_eager_bitdiff": int((r2.detach() != r.detach()).sum().item()),
            "compiled_vs_eager_maxabs": float(ae[finite].max().item()) if finite.any() else float("nan"),
            "compiled_vs_eager_bitdiff": int((g.detach() != r.detach()).sum().item()),
            "nonfinite": int((~finite).sum().item()),
            "matched_ratio_k1": mr(g, r, atol, rtol),
        }
        k = 1; need = None
        while k <= 2 ** 20:
            if mr(g, r, atol * k, rtol * k) >= 0.99:
                need = k; break
            k *= 2
        row["min_k_to_pass"] = need
        rows.append(row)
    doc = {"problem": problem_key(prob), "uuid": a.uuid, "mode": a.mode,
           "axes": dict(w.axes), "atol": atol, "rtol": rtol, "max_error_cap": cap,
           "torch": torch.__version__,
           "dynamo_recompile_limit": getattr(__import__("torch._dynamo", fromlist=["config"]).config, "recompile_limit", None),
           "outputs": rows}
    print(json.dumps(doc, indent=1))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
