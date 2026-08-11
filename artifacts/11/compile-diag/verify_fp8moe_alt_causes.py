#!/usr/bin/env python3
"""Adversarial alternative-cause probe for Quant__004_fp8_moe_expert_linear.

Independent of the claiming agent's scripts. Uses the harness input path.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path

sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")

from _common import exec_reference, load_problem, prepare_inputs  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda:0"
PROB = Path("/work/data/SOL-ExecBench/benchmark/Quant/004_fp8_moe_expert_linear")
UUID = "31856bae-d378-581b-9b0e-cdc02fbabe56"

ORIG_LINE = "    gated_output = F.silu(gate) * up  # SiLU activation on gate, element-wise multiply"

OPAQUE_PRELUDE = '''
import torch as _t
@_t.library.custom_op("advprobe::silu_opaque", mutates_args=())
def _silu_opaque(x: _t.Tensor) -> _t.Tensor:
    return _t.nn.functional.silu(x)
@_silu_opaque.register_fake
def _(x):
    return _t.empty_like(x)
'''


def diffstats(a, b):
    x = a.detach().to(torch.float32)
    y = b.detach().to(torch.float32)
    d = (x - y).abs()
    n = int((d > 0).sum().item())
    return {
        "n_diff": n,
        "numel": d.numel(),
        "frac": n / d.numel(),
        "max_abs": float(d.max().item()),
        "bit_identical": n == 0,
    }


def build(src, mode):
    ns = {}
    exec(compile(src, "<ref>", "exec"), ns)
    fn = ns["run"]
    if mode is None:
        return fn, ns
    kw = {} if mode == "default" else {"mode": mode}
    return torch.compile(fn, dynamic=False, **kw), ns


def run_pair(src_e, src_c, definition, w, label, mode="default"):
    eager, ns_e = build(src_e, None)
    comp, ns_c = build(src_c, mode)
    torch.manual_seed(0)
    ins_e = prepare_inputs(definition, w, ns_e, device=DEV)
    out_e = eager(*ins_e)
    torch.cuda.synchronize()
    torch.manual_seed(0)
    ins_c = prepare_inputs(definition, w, ns_c, device=DEV)
    out_c = comp(*ins_c)
    torch.cuda.synchronize()
    s = diffstats(out_c, out_e)
    print(f"[{label}] mode={mode} n_diff={s['n_diff']}/{s['numel']} "
          f"frac={s['frac']:.6f} max_abs={s['max_abs']:.6e} "
          f"bit_identical={s['bit_identical']}", flush=True)
    return s, out_e, out_c, ins_e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    a = ap.parse_args()

    definition, workloads = load_problem(PROB)
    w = [x for x in workloads if x.uuid == UUID][0]
    src = definition.reference
    assert ORIG_LINE in src, "reference line not found"

    print("torch:", torch.__version__, "dev:", torch.cuda.get_device_name(0))
    print("allow_tf32 (matmul):", torch.backends.cuda.matmul.allow_tf32,
          "| cudnn.allow_tf32:", torch.backends.cudnn.allow_tf32,
          "| fp16_reduced:", torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
          "| bf16_reduced:", torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)

    if a.test == "baseline":
        s, out_e, out_c, ins = run_pair(src, src, definition, w, "T0-baseline")
        # input mutation check
        torch.manual_seed(0)
        eager2, ns2 = build(src, None)
        ins2 = prepare_inputs(definition, w, ns2, device=DEV)
        mut = {k: diffstats(v1, v2)["bit_identical"]
               for k, (v1, v2) in enumerate(zip(ins, ins2))}
        print("T0 inputs-unmutated-after-eager-run:", mut)

    elif a.test == "opaque":
        # Only the COMPILED side gets the opaque op; eager side keeps F.silu.
        # The custom op is numerically identical to F.silu in eager, so any
        # residual divergence is NOT the elided rounding.
        new_line = "    gated_output = torch.ops.advprobe.silu_opaque(gate) * up"
        src_op = OPAQUE_PRELUDE + src.replace(ORIG_LINE, new_line)
        # sanity: eager(opaque src) must equal eager(orig src) bit-exactly
        e_orig, ns1 = build(src, None)
        e_op, ns2 = build(src_op, None)
        torch.manual_seed(0); i1 = prepare_inputs(definition, w, ns1, device=DEV)
        o1 = e_orig(*i1); torch.cuda.synchronize()
        torch.manual_seed(0); i2 = prepare_inputs(definition, w, ns2, device=DEV)
        o2 = e_op(*i2); torch.cuda.synchronize()
        s0 = diffstats(o2, o1)
        print(f"[T1-sanity eager(opaque) vs eager(orig)] n_diff={s0['n_diff']} "
              f"max_abs={s0['max_abs']:.6e} bit_identical={s0['bit_identical']}")
        run_pair(src, src_op, definition, w, "T1-opaque-silu-compiled-vs-orig-eager")

    elif a.test == "fp32single":
        # Force a SINGLE rounding in BOTH eager and compiled.
        new_line = ("    gated_output = (F.silu(gate.float()) * up.float())"
                    ".to(torch.bfloat16)")
        src2 = src.replace(ORIG_LINE, new_line)
        run_pair(src2, src2, definition, w, "T2-fp32-single-round")

    elif a.test == "forceround":
        # Force TWO roundings explicitly in both (what eager already does).
        new_line = ("    gated_output = F.silu(gate).to(torch.bfloat16)"
                    ".to(torch.float32).to(torch.bfloat16) * up")
        src2 = src.replace(ORIG_LINE, new_line)
        run_pair(src, src2, definition, w, "T3-explicit-double-round-compiled")

    elif a.test == "isolate":
        # Feed bit-identical operands to isolated sub-computations.
        torch.manual_seed(0)
        _, ns = build(src, None)
        ins = prepare_inputs(definition, w, ns, device=DEV)
        hidden, routing, gate_up_w, down_w = ins
        BS = ns["BlockwiseScaler"]; ST = ns["ScalingType"]
        act = BS(ST.BlockWise1x128); wt = BS(ST.BlockWise128x128)

        # (a) fp32 GEMM in isolation -> tests TF32 / triton-template / accum
        hf = hidden.to(torch.float32)
        sh = act.compute_scales(hf)
        gw = gate_up_w.to(torch.float32).T
        sg = wt.compute_scales(gw)
        hs = act.apply_scaling(hf, sh, inverse=False, clamp_to_fp8_range=True)
        gs = wt.apply_scaling(gw, sg, inverse=False, clamp_to_fp8_range=True)
        hfp8 = hs.to(torch.float8_e4m3fn); gfp8 = gs.T.to(torch.float8_e4m3fn)
        a_f32 = act.apply_scaling(hfp8.to(torch.float32), sh, inverse=True)
        b_f32 = wt.apply_scaling(gfp8.to(torch.float32), sg.T.contiguous(), inverse=True)

        def gemm(x, y):
            return (x @ y.T).to(torch.bfloat16)
        cg = torch.compile(gemm, dynamic=False)
        r_e = gemm(a_f32, b_f32); r_c = cg(a_f32, b_f32)
        torch.cuda.synchronize()
        s = diffstats(r_c, r_e)
        print(f"[I1-fp32-gemm-isolated] n_diff={s['n_diff']}/{s['numel']} "
              f"max_abs={s['max_abs']:.6e} bit_identical={s['bit_identical']}")

        # (b) fp8 quantize chain in isolation -> tests cast rounding / clamp / amax
        def quant(t):
            f = t.to(torch.float32)
            s_ = act.compute_scales(f)
            sc = act.apply_scaling(f, s_, inverse=False, clamp_to_fp8_range=True)
            return sc.to(torch.float8_e4m3fn).view(torch.uint8), s_
        cq = torch.compile(quant, dynamic=False)
        q_e = quant(hidden); q_c = cq(hidden)
        torch.cuda.synchronize()
        sb = diffstats(q_c[0].to(torch.int32), q_e[0].to(torch.int32))
        ss = diffstats(q_c[1], q_e[1])
        print(f"[I2-fp8-quant-codes] n_diff={sb['n_diff']}/{sb['numel']} "
              f"bit_identical={sb['bit_identical']}")
        print(f"[I2-fp8-quant-scales] n_diff={ss['n_diff']}/{ss['numel']} "
              f"bit_identical={ss['bit_identical']}")

        # (c) silu*up in isolation on real bf16 operands
        gate_up = ns["CuBLASRefBlockwiseGemm"]().scaled_mm(
            mat_a=hfp8, mat_b=gfp8, scale_a=sh,
            scale_recipe_a=ST.BlockWise1x128,
            scale_b=sg.T.contiguous(), scale_recipe_b=ST.BlockWise128x128,
            bias=None, output_dtype=torch.bfloat16)
        g, u = gate_up.chunk(2, dim=-1)

        def sil(x, y):
            return F.silu(x) * y
        cs = torch.compile(sil, dynamic=False)
        z_e = sil(g, u); z_c = cs(g, u)
        torch.cuda.synchronize()
        s3 = diffstats(z_c, z_e)
        print(f"[I3-silu-mul-isolated] n_diff={s3['n_diff']}/{s3['numel']} "
              f"frac={s3['frac']:.6f} max_abs={s3['max_abs']:.6e}")
        # is eager exactly "round silu to bf16, then mul"?
        z_ref2 = (F.silu(g.float()).to(torch.bfloat16)) * u
        s4 = diffstats(z_ref2, z_e)
        print(f"[I3b-eager==bf16round(silu)*up] n_diff={s4['n_diff']} "
              f"bit_identical={s4['bit_identical']}")
        # is compiled exactly "single rounding in fp32"?
        z_ref3 = (F.silu(g.float()) * u.float()).to(torch.bfloat16)
        s5 = diffstats(z_ref3, z_c)
        print(f"[I3c-compiled==fp32single] n_diff={s5['n_diff']}/{s5['numel']} "
              f"max_abs={s5['max_abs']:.6e} bit_identical={s5['bit_identical']}")

    else:
        raise SystemExit("unknown test")


if __name__ == "__main__":
    main()
