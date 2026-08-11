#!/usr/bin/env python3
"""Op-level eager-vs-Inductor fidelity, to separate 'bf16 upcast policy' from
'different fp32 transcendental / scan implementation' as causes."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda:0"
print("# torch", torch.__version__, torch.cuda.get_device_name(0))


def bitdist(a, b):
    """max |bitpattern distance| between two same-dtype tensors, and frac exact."""
    if a.dtype == torch.bfloat16:
        ia = a.view(torch.int16).to(torch.int32)
        ib = b.view(torch.int16).to(torch.int32)
    elif a.dtype == torch.float32:
        ia = a.view(torch.int32).to(torch.int64)
        ib = b.view(torch.int32).to(torch.int64)
    else:
        raise TypeError(a.dtype)
    # map to monotonic ordering
    def mono(i):
        return torch.where(i < 0, torch.iinfo(i.dtype).min - i, i)
    d = (mono(ia) - mono(ib)).abs()
    return int(d.max().item()), float((d == 0).float().mean().item())


OPS = {
    "sigmoid": lambda t: torch.sigmoid(t),
    "exp": lambda t: torch.exp(t),
    "softplus": lambda t: F.softplus(t),
    "rsqrt": lambda t: torch.rsqrt(t.abs() + 1e-5),
    "silu_chain": lambda t: t * torch.sigmoid(t),
    "cumsum_last": lambda t: torch.cumsum(t, dim=-1),
    "mean_last": lambda t: t.float().pow(2).mean(dim=-1, keepdim=True),
}

print("\n== per-op eager vs compiled, same input ==")
print(f"{'op':12s} {'dtype':9s} {'ulpdist':>8s} {'frac_exact':>11s}  "
      f"{'vs fp32-then-round: ulpdist':>28s} {'exact':>7s}")
for dtype in (torch.float32, torch.bfloat16):
    for name, fn in OPS.items():
        x = (torch.randn(64, 4096, device=dev, dtype=torch.float32) * 3).to(dtype)
        ref = fn(x)
        cf = torch.compile(fn, dynamic=False)
        got = cf(x)
        torch.cuda.synchronize()
        if ref.dtype != got.dtype:
            print(f"{name:12s} {str(dtype)[6:]:9s} DTYPE MISMATCH {ref.dtype} vs {got.dtype}")
            continue
        d, e = bitdist(got, ref)
        extra = ""
        if dtype == torch.bfloat16 and ref.dtype == torch.bfloat16:
            f32 = fn(x.float()).to(torch.bfloat16)
            d2, e2 = bitdist(got, f32)
            extra = f"{d2:>28d} {e2:>7.4f}"
        print(f"{name:12s} {str(dtype)[6:]:9s} {d:>8d} {e:>11.6f}  {extra}")
        del x, ref, got
        torch.cuda.empty_cache()

print("\n== library ops (matmul / conv1d) eager vs compiled ==")
b, s, h = 1, 512, 8192
proj = 37120
hs = torch.randn(b, s, h, device=dev, dtype=torch.bfloat16)
w = torch.randn(proj, h, device=dev, dtype=torch.bfloat16) * 0.02
mm = lambda a, bb: torch.matmul(a, bb.t())
r = mm(hs, w); g = torch.compile(mm, dynamic=False)(hs, w); torch.cuda.synchronize()
print("matmul bf16:", bitdist(g, r))
del r, g, hs, w
torch.cuda.empty_cache()

conv_dim = 16384 + 2 * 2048
x = torch.randn(b, conv_dim, s, device=dev, dtype=torch.bfloat16)
cw = torch.randn(conv_dim, 1, 4, device=dev, dtype=torch.bfloat16)
cb = torch.randn(conv_dim, device=dev, dtype=torch.bfloat16)
cv = lambda a, ww, bb: F.conv1d(a, ww, bb, padding=3, groups=conv_dim)[..., :s]
r = cv(x, cw, cb); g = torch.compile(cv, dynamic=False)(x, cw, cb); torch.cuda.synchronize()
print("conv1d depthwise bf16:", bitdist(g, r))
del r, g, x, cw, cb
torch.cuda.empty_cache()

print("\n== fp32 einsum eager vs compiled (shapes from the scan) ==")
B_, C_, L_, H_, N_ = 1, 4, 128, 256, 256
Cc = torch.randn(B_, C_, L_, H_, N_, device=dev)
Bc = torch.randn(B_, C_, L_, H_, N_, device=dev)
es = lambda a, bb: torch.einsum('bclhn,bcshn->bclsh', a, bb)
r = es(Cc, Bc); g = torch.compile(es, dynamic=False)(Cc, Bc); torch.cuda.synchronize()
print("einsum G fp32:", bitdist(g, r))
