#!/usr/bin/env python3
"""Adversarial-verifier microtest: is Inductor's bf16 pointwise upcast
universal, and does it by itself decide PASS/FAIL?

Runs the exact claimed mechanism in isolation:
    y = x * sigmoid(x)     with x bfloat16
eager vs torch.compile, plus the two hypothesised bit-references
(all-bf16 intermediates, fp32 intermediates rounded once).
"""
import torch

torch.manual_seed(0)


def silu_mul(x):
    return x * torch.sigmoid(x)


def report(tag, x):
    eager = silu_mul(x)
    c = torch.compile(silu_mul, dynamic=False)
    comp = c(x)
    torch.cuda.synchronize()
    bf16_ref = (x * torch.sigmoid(x))                       # bf16 intermediates
    fp32_ref = (x.float() * torch.sigmoid(x.float())).to(x.dtype)
    d = (comp.float() - eager.float()).abs()
    print(f"[{tag}] dtype={x.dtype} n={x.numel()} absmax={float(x.abs().max()):.4g}")
    print(f"   compiled-vs-eager   max_abs={float(d.max()):.6e}  "
          f"frac_diff={float((d > 0).float().mean()):.4f}")
    print(f"   eager == bf16_ref   {bool((eager == bf16_ref).all())}")
    print(f"   compiled == fp32_ref {bool((comp == fp32_ref).all())}")
    print(f"   compiled == bf16_ref {bool((comp == bf16_ref).all())}")
    print(f"   bf16_ref vs fp32_ref max_abs="
          f"{float((bf16_ref.float() - fp32_ref.float()).abs().max()):.6e} "
          f"frac={float(((bf16_ref != fp32_ref)).float().mean()):.4f}")


for scale, tag in ((10.25, "scale~10 (058 conv_out scale)"), (1.0, "scale~1")):
    x = (torch.randn(4096, 512, device="cuda", dtype=torch.float32) * scale).to(torch.bfloat16)
    report(tag, x)
    torch._dynamo.reset()
