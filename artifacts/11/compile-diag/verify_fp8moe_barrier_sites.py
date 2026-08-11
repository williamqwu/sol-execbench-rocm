#!/usr/bin/env python3
"""Adversarial probe 2: which Inductor elision(s) actually cause the divergence?

Inserts opaque `materialize` barriers at candidate sites in the COMPILED
source only. Eager source is untouched. A barrier is numerically a no-op in
eager, so any gap that closes when a barrier is added is caused by the
elision at that site.
"""
from __future__ import annotations
import sys, argparse, itertools
from pathlib import Path

sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")
from _common import exec_reference, load_problem, prepare_inputs  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda:0"
PROB = Path("/work/data/SOL-ExecBench/benchmark/Quant/004_fp8_moe_expert_linear")
UUID = "31856bae-d378-581b-9b0e-cdc02fbabe56"

PRELUDE = '''
import torch as _t
@_t.library.custom_op("advp2::mat", mutates_args=())
def _mat(x: _t.Tensor) -> _t.Tensor:
    return x.clone()
@_mat.register_fake
def _(x):
    return _t.empty_like(x)
@_t.library.custom_op("advp2::silu_op", mutates_args=())
def _silu_op(x: _t.Tensor) -> _t.Tensor:
    return _t.nn.functional.silu(x)
@_silu_op.register_fake
def _(x):
    return _t.empty_like(x)
'''

A_SILU = "    gated_output = F.silu(gate) * up  # SiLU activation on gate, element-wise multiply"
A_CHUNK = "    gate, up = gate_up_output.chunk(2, dim=-1)"
A_STEP4 = "    # Step 4: Apply routing weight (NOT quantized)"
A_GFP32 = "    gated_fp32 = gated_output.to(torch.float32)"


def patch(src, sites):
    s = src
    if "gemm1" in sites:
        s = s.replace(A_CHUNK,
                      "    gate_up_output = torch.ops.advp2.mat(gate_up_output)\n" + A_CHUNK)
    if "silu" in sites:
        s = s.replace(A_SILU, "    gated_output = torch.ops.advp2.silu_op(gate) * up")
    if "gated" in sites:
        s = s.replace(A_GFP32,
                      "    gated_output = torch.ops.advp2.mat(gated_output)\n" + A_GFP32)
    if "gemm2" in sites:
        s = s.replace(A_STEP4, "    output = torch.ops.advp2.mat(output)\n" + A_STEP4)
    return PRELUDE + s


def diffstats(a, b):
    x = a.detach().to(torch.float32); y = b.detach().to(torch.float32)
    d = (x - y).abs(); n = int((d > 0).sum().item())
    return n, d.numel(), float(d.max().item()), float(d.mean().item())


def build(src, mode):
    ns = {}
    exec(compile(src, "<ref>", "exec"), ns)
    fn = ns["run"]
    return (fn if mode is None else torch.compile(fn, dynamic=False)), ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="")
    a = ap.parse_args()
    sites = set(x for x in a.sites.split(",") if x)

    definition, workloads = load_problem(PROB)
    w = [x for x in workloads if x.uuid == UUID][0]
    src = definition.reference
    for anc in (A_SILU, A_CHUNK, A_STEP4, A_GFP32):
        assert anc in src, anc

    eager, ns_e = build(src, None)
    torch.manual_seed(0); ins_e = prepare_inputs(definition, w, ns_e, device=DEV)
    out_e = eager(*ins_e); torch.cuda.synchronize()

    src_c = patch(src, sites)
    comp, ns_c = build(src_c, "default")
    torch.manual_seed(0); ins_c = prepare_inputs(definition, w, ns_c, device=DEV)
    out_c = comp(*ins_c); torch.cuda.synchronize()

    n, tot, mx, mean = diffstats(out_c, out_e)
    atol = float(w.tolerance.max_atol); rtol = float(w.tolerance.max_rtol)
    x = out_c.to(torch.float32); y = out_e.to(torch.float32)
    bad = ((x - y).abs() > (atol + rtol * y.abs()))
    mr = 1.0 - float(bad.sum().item()) / bad.numel()
    print(f"sites={sorted(sites) or ['NONE']} n_diff={n}/{tot} frac={n/tot:.6f} "
          f"max_abs={mx:.6e} mean_abs={mean:.6e} matched_ratio={mr:.6f} "
          f"-> {'PASS' if mr >= 0.99 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
