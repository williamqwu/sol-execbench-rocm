#!/usr/bin/env python3
"""L1__062 deep diagnostic. Per-workload fresh compile (dynamo reset), per-output
breakdown, fp32-no-intermediate-rounding emulation, fp64 CPU golden.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")

from _common import exec_reference, load_problem, prepare_inputs, problem_key  # noqa: E402

import torch  # noqa: E402

OUT_NAMES = ["grad_key_states", "grad_value_states", "grad_cos", "grad_sin",
             "grad_key_cache_input", "grad_value_cache_input"]

MODES = {"eager": None, "default": "__default__",
         "max-autotune": "max-autotune-no-cudagraphs"}


def build(ref_src, mode):
    ns = {}
    exec(compile(ref_src, "<reference>", "exec"), ns)
    fn = ns["run"]
    if mode is None:
        return fn, ns
    kw = {} if mode == "__default__" else {"mode": mode}
    return torch.compile(fn, dynamic=False, **kw), ns


def per_output(got, ref, atol, rtol):
    rows = []
    for name, g, r in zip(OUT_NAMES, got, ref):
        x = g.detach().to(torch.float32)
        y = r.detach().to(torch.float32)
        ae = (x - y).abs()
        bad = ae > (atol + rtol * y.abs())
        n = ae.numel()
        nbad = int(bad.sum().item())
        rows.append({
            "output": name, "shape": list(g.shape), "numel": n,
            "max_abs": float(ae.max().item()),
            "max_rel": float((ae / y.abs().clamp(min=atol)).max().item()),
            "n_mismatch": nbad,
            "matched_ratio": 1.0 - nbad / n,
            "pass": (1.0 - nbad / n) >= 0.99,
            # what the reference value looks like where it mismatches
            "median_abs_ref_at_mismatch": (
                float(y.abs()[bad].median().item()) if nbad else None),
            "median_abs_ref_all": float(y.abs().median().item()),
        })
    return rows


# ---------- fp32-throughout emulation (no bf16 intermediate rounding) --------
def run_fp32_nointermediate(gkc, gvc, ks, cos, sin, cp):
    """Same math, every tensor promoted to fp32 up front: no bf16 rounding of
    any intermediate. Cast to bf16 only at the very end."""
    gkc32, gvc32, ks32 = gkc.float(), gvc.float(), ks.float()
    cos32, sin32 = cos.float(), sin.float()
    h = ks.shape[-1] // 2
    k1, k2 = ks32[..., :h], ks32[..., h:]
    krh = torch.cat((-k2, k1), dim=-1)
    ce, se = cos32.unsqueeze(1), sin32.unsqueeze(1)
    gksr = gkc32[:, :, cp]
    gvs = gvc32[:, :, cp]
    a = gksr * ce
    b = gksr * se
    g1 = a[..., :h] + b[..., h:]
    g2 = a[..., h:] - b[..., :h]
    gks = torch.cat([g1, g2], dim=-1)
    gcos = (gksr * ks32).sum(dim=1)
    gsin = (gksr * krh).sum(dim=1)
    gkci = gkc32.clone(); gkci[:, :, cp] = 0
    gvci = gvc32.clone(); gvci[:, :, cp] = 0
    return (gks, gvs, gcos, gsin, gkci, gvci)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--mode", default="default", choices=list(MODES))
    ap.add_argument("--uuid", action="append", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()

    prob = Path(a.problem).resolve()
    definition, workloads = load_problem(prob)
    if a.uuid:
        workloads = [w for w in workloads if w.uuid in set(a.uuid)]

    ref_run, ref_ns = exec_reference(definition)
    rows = []
    for w in workloads:
        atol = float(w.tolerance.max_atol); rtol = float(w.tolerance.max_rtol)
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            ins = prepare_inputs(definition, w, ref_ns, device="cuda:0")
            out_ref = ref_run(*ins)
            torch.cuda.synchronize()

            # eager vs eager, same seed, fresh inputs
            torch.manual_seed(seed)
            ins2 = prepare_inputs(definition, w, ref_ns, device="cuda:0")
            out_ref2 = ref_run(*ins2)
            torch.cuda.synchronize()
            e2e = per_output(out_ref2, out_ref, atol, rtol)

            # FRESH compile for this shape
            torch._dynamo.reset()
            cmp_run, cmp_ns = build(definition.reference, MODES[a.mode])
            torch.manual_seed(seed)
            ins3 = prepare_inputs(definition, w, cmp_ns, device="cuda:0")
            out_cmp = cmp_run(*ins3)
            torch.cuda.synchronize()
            c2e = per_output(out_cmp, out_ref, atol, rtol)

            mr = min(r["matched_ratio"] for r in c2e)
            verdict = "PASS" if all(r["pass"] for r in c2e) else "FAIL"
            mr_e = min(r["matched_ratio"] for r in e2e)
            rows.append({
                "uuid": w.uuid, "axes": dict(w.axes), "seed": seed,
                "tol_atol": atol, "tol_rtol": rtol,
                "eager_vs_eager_max_abs": max(r["max_abs"] for r in e2e),
                "eager_vs_eager_mr": mr_e,
                "compiled_vs_eager_max_abs": max(r["max_abs"] for r in c2e),
                "compiled_vs_eager_max_rel": max(r["max_rel"] for r in c2e),
                "compiled_vs_eager_mr": mr,
                "verdict": verdict,
                "per_output": c2e,
            })
            worst = min(c2e, key=lambda r: r["matched_ratio"])
            print(f"{w.uuid[:8]} s{seed} {str(dict(w.axes))[:46]:46s} "
                  f"atol={atol:.3e} | e/e={rows[-1]['eager_vs_eager_max_abs']:.3e} "
                  f"| c/e={rows[-1]['compiled_vs_eager_max_abs']:.3e} "
                  f"rel={rows[-1]['compiled_vs_eager_max_rel']:.3e} "
                  f"mr={mr:.6f} worst={worst['output']}({worst['numel']}) "
                  f"nbad={worst['n_mismatch']} -> {verdict}", flush=True)
            del ins, ins2, ins3, out_ref, out_ref2, out_cmp
            torch.cuda.empty_cache()

    doc = {"problem": problem_key(prob), "mode": a.mode,
           "torch": torch.__version__,
           "note": "fresh torch._dynamo.reset() + torch.compile per workload",
           "rows": rows}
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
