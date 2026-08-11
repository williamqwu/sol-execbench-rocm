#!/usr/bin/env python3
"""Adversarial check of alternative causes for L2__058 compile divergence."""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path

ROOT = Path("/work")
sys.path.insert(0, str(ROOT / "scripts" / "runners"))
sys.path.insert(0, str(ROOT / "src"))
from _common import exec_reference, load_problem, prepare_inputs  # noqa

import torch

PROB = ROOT / "data/SOL-ExecBench/benchmark/L2/058_mamba2_selective_scan"
UUID = "4c88b9e7-bf31-5344-9ca7-10eb7e3242e4"

SILU_ORIG = "hidden_B_C = (conv_out * torch.sigmoid(conv_out)).transpose(1, 2)  # silu"
SILU_F32 = ("_c32 = conv_out.float()\n    hidden_B_C = "
            "(_c32 * torch.sigmoid(_c32)).to(conv_out.dtype).transpose(1, 2)  # silu")
SILU_F32_KEEP = ("_c32 = conv_out.float()\n    hidden_B_C = "
                 "(_c32 * torch.sigmoid(_c32)).transpose(1, 2)  # silu fp32 kept")
SP_ORIG = "dt = F.softplus(dt + dt_bias)"
SP_F32 = "dt = F.softplus(dt.float() + dt_bias.float()).to(dt.dtype)"
SP_F32_KEEP = "dt = F.softplus(dt.float() + dt_bias.float())"
GATE_ORIG = "y = y_normed * (gate * torch.sigmoid(gate))  # silu gate"
GATE_F32 = ("_g32 = gate.float()\n    y = (y_normed.float() * (_g32 * torch.sigmoid(_g32)))"
            ".to(y_normed.dtype)  # silu gate")


def variant(src, *subs):
    out = src
    for a, b in subs:
        assert a in out, a[:40]
        out = out.replace(a, b)
    return out


def make_run(src):
    ns = {}
    exec(compile(src, "<v>", "exec"), ns)
    return ns["run"], ns


def as_t(o):
    return o if isinstance(o, torch.Tensor) else o[0]


def cmp(a, b, atol, rtol, label=""):
    x = as_t(a).detach().float()
    y = as_t(b).detach().float()
    d = (x - y).abs()
    bad = d > (atol + rtol * y.abs())
    mr = 1.0 - bad.float().mean().item()
    exact = (as_t(a).detach() == as_t(b).detach()).float().mean().item()
    return dict(label=label, max_abs=d.max().item(), mean_abs=d.mean().item(),
                matched_ratio=mr, frac_exact=exact,
                verdict="PASS" if mr >= 0.99 else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuid", default=UUID)
    a = ap.parse_args()

    definition, workloads = load_problem(PROB)
    wl = [w for w in workloads if w.uuid.startswith(a.uuid[:8])][0]
    atol = float(wl.tolerance.max_atol); rtol = float(wl.tolerance.max_rtol)
    print(f"# workload {wl.uuid[:8]} axes={dict(wl.axes)} atol={atol:.4e} rtol={rtol:.4e}")
    print(f"# torch {torch.__version__} dev {torch.cuda.get_device_name(0)}")
    print("# backend flags:",
          "allow_tf32=", torch.backends.cuda.matmul.allow_tf32,
          "fp32_precision=", torch.get_float32_matmul_precision(),
          "bf16_reduced=", torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
          "fp16_reduced=", torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
          "cudnn.allow_tf32=", torch.backends.cudnn.allow_tf32)
    import torch._inductor.config as ic
    print("# inductor: emulate_precision_casts=", getattr(ic, "emulate_precision_casts", "n/a"),
          "fallback_random=", ic.fallback_random,
          "triton.cudagraphs=", ic.triton.cudagraphs)

    src = definition.reference
    run_eager, ns = exec_reference(definition)

    torch.manual_seed(0)
    ins = prepare_inputs(definition, wl, ns, device="cuda:0")

    def sig(ts):
        return [float(t.float().abs().sum().item()) if isinstance(t, torch.Tensor) else t
                for t in ts]
    s0 = sig(ins)
    out_e = run_eager(*ins); torch.cuda.synchronize()
    s1 = sig(ins)
    print("A. inputs mutated by eager:", s0 != s1)

    out_e2 = run_eager(*ins); torch.cuda.synchronize()
    print("A2. eager determinism (same inputs, 2 calls) max_abs:",
          cmp(out_e2, out_e, atol, rtol)["max_abs"])
    del out_e2

    cfn = torch.compile(make_run(src)[0], dynamic=False)
    out_c = cfn(*ins); torch.cuda.synchronize()
    s2 = sig(ins)
    print("A3. inputs mutated by compiled:", s1 != s2)
    out_c2 = cfn(*ins); torch.cuda.synchronize()
    print("A4. compiled determinism max_abs:", cmp(out_c2, out_c, atol, rtol)["max_abs"])
    del out_c2

    base = cmp(out_c, out_e, atol, rtol, "compiled vs eager")
    print("B. BASELINE", json.dumps(base))

    x = as_t(out_c); y = as_t(out_e)
    yf = y.float().abs()
    ulp = torch.where(yf > 0, yf.log2().floor().exp2() * (2 ** -8),
                      torch.full_like(yf, 2.0 ** -133))
    d = (x.float() - y.float()).abs()
    r = d / ulp
    print("C. diff/ULP(bf16 at ref magnitude): max=%.3f mean=%.4f frac>1.5=%.3e frac>0=%.4f"
          % (r.max().item(), r.mean().item(), (r > 1.5).float().mean().item(),
             (r > 0).float().mean().item()))
    del r, d, ulp, yf
    torch.cuda.empty_cache()

    import torch.nn.functional as F
    hs, in_w, cw, cb = ins[0], ins[1], ins[2], ins[3]
    projected = torch.matmul(hs, in_w.t())
    inter, gts, nh = 16384, 8 * 256, 256
    conv_dim = inter + 2 * gts
    gs = projected.shape[-1] - inter - conv_dim - nh
    hbc = projected[..., gs + inter: gs + inter + conv_dim]
    seq_len = hs.shape[1]
    conv_out = F.conv1d(hbc.transpose(1, 2), cw, cb, padding=3, groups=conv_dim)[..., :seq_len]
    silu_bf16 = conv_out * torch.sigmoid(conv_out)
    c32 = conv_out.float()
    silu_f32 = (c32 * torch.sigmoid(c32)).to(conv_out.dtype)
    dd = (silu_bf16.float() - silu_f32.float()).abs()
    print("D. eager-bf16-SiLU vs fp32-SiLU: max_abs=%.6e frac_differ=%.4f conv_absmax=%.4e"
          % (dd.max().item(), (dd > 0).float().mean().item(),
             conv_out.abs().max().float().item()))
    del projected, hbc, conv_out, silu_bf16, silu_f32, c32, dd
    torch.cuda.empty_cache()

    variants = {
        "V1 silu fp32(rounded)": [(SILU_ORIG, SILU_F32)],
        "V2 silu fp32(kept)": [(SILU_ORIG, SILU_F32_KEEP)],
        "V3 silu+sp fp32(kept)": [(SILU_ORIG, SILU_F32_KEEP), (SP_ORIG, SP_F32_KEEP)],
        "V4 silu+sp+gate fp32": [(SILU_ORIG, SILU_F32_KEEP), (SP_ORIG, SP_F32_KEEP),
                                 (GATE_ORIG, GATE_F32)],
        "V5 sp fp32 only": [(SP_ORIG, SP_F32_KEEP)],
        "V6 gate fp32 only": [(GATE_ORIG, GATE_F32)],
    }
    for name, subs in variants.items():
        vr, vns = make_run(variant(src, *subs))
        try:
            ov = vr(*ins); torch.cuda.synchronize()
        except Exception as e:
            print(f"E. {name}: ERROR {type(e).__name__}: {e}")
            continue
        vs_c = cmp(ov, out_c, atol, rtol)
        vs_e = cmp(ov, out_e, atol, rtol)
        print("E. %-24s vs_compiled max_abs=%.4e exact=%.4f mr=%.6f | "
              "vs_eager max_abs=%.4e exact=%.4f mr=%.6f %s"
              % (name, vs_c["max_abs"], vs_c["frac_exact"], vs_c["matched_ratio"],
                 vs_e["max_abs"], vs_e["frac_exact"], vs_e["matched_ratio"], vs_e["verdict"]))
        del ov
        torch.cuda.empty_cache()

    for flag in [True, False]:
        torch.backends.cuda.matmul.allow_tf32 = flag
        torch.backends.cudnn.allow_tf32 = flag
        o = run_eager(*ins); torch.cuda.synchronize()
        c = cmp(o, out_e, atol, rtol)
        print("F. eager allow_tf32=%s vs baseline eager: max_abs=%.4e exact=%.4f"
              % (flag, c["max_abs"], c["frac_exact"]))
        del o
        torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for flag in [False, True]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
        o = run_eager(*ins); torch.cuda.synchronize()
        c = cmp(o, out_e, atol, rtol)
        print("F2. eager bf16_reduced=%s vs baseline eager: max_abs=%.4e exact=%.4f"
              % (flag, c["max_abs"], c["frac_exact"]))
        del o
        torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


if __name__ == "__main__":
    raise SystemExit(main())
