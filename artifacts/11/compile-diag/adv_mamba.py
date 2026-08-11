#!/usr/bin/env python3
"""ADVERSARIAL check: is the compiled-vs-eager divergence caused by the
compiler, or by the two runs receiving different inputs?

Strategy: eliminate input generation from the experiment entirely. Generate
ONE set of inputs, hash every byte, then feed byte-identical copies to eager
and to the compiled callable. Any residual difference is the compiler's.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")

from _common import exec_reference, load_problem, prepare_inputs  # noqa: E402
import torch  # noqa: E402


def thash(t):
    if not isinstance(t, torch.Tensor):
        return repr(t)
    b = t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def hashes(ins):
    return [thash(x) for x in ins]


def as_tuple(o):
    if isinstance(o, (list, tuple)):
        return tuple(o)
    if isinstance(o, dict):
        return tuple(o[k] for k in sorted(o))
    return (o,)


def stats(got, ref, atol, rtol):
    x = got.detach().to(torch.float32)
    y = ref.detach().to(torch.float32)
    ae = (x - y).abs()
    bad = (ae > (atol + rtol * y.abs())) | ~torch.isfinite(ae)
    mr = 1.0 - float(bad.sum().item()) / float(ae.numel())
    return {"max_abs": float(ae.max().item()), "matched_ratio": mr,
            "frac_diff": float((x != y).float().mean().item())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="/work/data/SOL-ExecBench/benchmark/L2/058_mamba2_selective_scan")
    ap.add_argument("--uuid", required=True)
    ap.add_argument("--mode", default="default")
    a = ap.parse_args()

    prob = Path(a.problem)
    definition, workloads = load_problem(prob)
    w = [x for x in workloads if x.uuid.startswith(a.uuid)][0]
    atol = float(w.tolerance.max_atol); rtol = float(w.tolerance.max_rtol)
    print(f"uuid={w.uuid} axes={dict(w.axes)} atol={atol:.6e} rtol={rtol:.6e} "
          f"req_mr={w.tolerance.required_matched_ratio}")

    ref_run, ref_ns = exec_reference(definition)
    ns2: dict = {}
    exec(compile(definition.reference, "<ref>", "exec"), ns2)
    raw2 = ns2["run"]

    # ---- 1. input determinism across namespaces, seeds and call order ------
    torch.manual_seed(0); insA = prepare_inputs(definition, w, ref_ns, device="cuda:0")
    torch.manual_seed(0); insB = prepare_inputs(definition, w, ns2, device="cuda:0")
    hA, hB = hashes(insA), hashes(insB)
    print("input hashes ref_ns :", hA)
    print("input hashes cmp_ns :", hB)
    print("INPUTS IDENTICAL ACROSS NAMESPACES:", hA == hB)

    # generate again after burning RNG, to confirm manual_seed(0) fully resets
    _ = torch.randn(1000, device="cuda:0")
    torch.manual_seed(0); insC = prepare_inputs(definition, w, ref_ns, device="cuda:0")
    print("INPUTS IDENTICAL AFTER RNG BURN:", hashes(insC) == hA)

    # ---- 2. eager determinism on BYTE-IDENTICAL inputs ---------------------
    out_ref = as_tuple(ref_run(*insA))[0]
    torch.cuda.synchronize()
    out_ref2 = as_tuple(ref_run(*insA))[0]   # same tensor objects, no regen
    torch.cuda.synchronize()
    s = stats(out_ref2, out_ref, atol, rtol)
    print(f"EAGER twice on THE SAME input objects: max_abs={s['max_abs']:.6e} "
          f"frac_diff={s['frac_diff']:.4f}")

    # ---- 3. compiled on THE SAME input objects -----------------------------
    kw = {} if a.mode == "default" else {"mode": a.mode}
    cfn = torch.compile(raw2, dynamic=False, **kw)
    out_cmp = as_tuple(cfn(*insA))[0]
    torch.cuda.synchronize()
    print("input hashes AFTER compiled call:", hashes(insA) == hA, "(inputs unmutated)")
    s = stats(out_cmp, out_ref, atol, rtol)
    print(f"COMPILED vs EAGER, SAME input objects: max_abs={s['max_abs']:.6e} "
          f"mr={s['matched_ratio']:.6f} frac_diff={s['frac_diff']:.4f} "
          f"-> {'FAIL' if s['matched_ratio'] < 0.99 else 'PASS'}")

    # compiled twice -> compiled determinism
    out_cmp2 = as_tuple(cfn(*insA))[0]
    torch.cuda.synchronize()
    s2 = stats(out_cmp2, out_cmp, atol, rtol)
    print(f"COMPILED twice, same inputs: max_abs={s2['max_abs']:.6e}")

    # ---- 4. compiled on FRESHLY REGENERATED inputs (diag script's path) ----
    torch.manual_seed(0); insD = prepare_inputs(definition, w, ns2, device="cuda:0")
    out_cmp3 = as_tuple(cfn(*insD))[0]
    torch.cuda.synchronize()
    s3 = stats(out_cmp3, out_ref, atol, rtol)
    print(f"COMPILED vs EAGER, REGENERATED inputs: max_abs={s3['max_abs']:.6e} "
          f"mr={s3['matched_ratio']:.6f}")
    print("regenerated == original inputs:", hashes(insD) == hA)


if __name__ == "__main__":
    main()
