#!/usr/bin/env python3
"""Per-stage isolation: compile each stage alone, feed IDENTICAL eager inputs."""
import sys
from pathlib import Path
sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")
import torch
import torch.nn.functional as F
from _common import exec_reference, load_problem, prepare_inputs

PROB = Path("/work/data/SOL-ExecBench/benchmark/L1/067_flash_attention_gqa_ultralong")
UUID = sys.argv[1] if len(sys.argv) > 1 else "b0c05812-9ac0-5ecb-a7a9-73edaf552dde"
H, KVH, HD, G = 32, 8, 128, 4
SCALING = HD ** -0.5

definition, workloads = load_problem(PROB)
w = [x for x in workloads if x.uuid == UUID][0]
_, ref_ns = exec_reference(definition)
torch.manual_seed(0)
hs, cos, sin, qw, kw, vw, ow = prepare_inputs(definition, w, ref_ns, device="cuda:0")
B, S, _ = hs.shape


def s_proj(hs, qw):
    return F.linear(hs, qw)


def s_rope(x, cos, sin):
    cos_e = cos.unsqueeze(1); sin_e = sin.unsqueeze(1)
    x1 = x[..., : HD // 2]; x2 = x[..., HD // 2:]
    rot = torch.cat((-x2, x1), dim=-1)
    return (x * cos_e) + (rot * sin_e)


def s_attn(q, k, v, seq_len):
    aw = torch.matmul(q, k.transpose(2, 3)) * SCALING
    cm = torch.triu(torch.full((seq_len, seq_len), float('-inf'),
                    device=q.device, dtype=q.dtype), diagonal=1)
    aw = aw + cm
    aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(aw, v)


def cmp(name, fn, *args):
    a = fn(*args)
    b = torch.compile(fn, dynamic=False)(*args)
    torch.cuda.synchronize()
    d = (a.float() - b.float()).abs()
    d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
    nd = int((d > 0).sum().item())
    print(f"{name:26s} max_abs={d.max().item():.4e}  n_diff={nd:9d} "
          f"frac={nd/a.numel():.4f}")
    return a, b


print(f"uuid={UUID} B={B} S={S}")
print("--- each stage compiled ALONE, given identical eager inputs ---")
q_e, _ = cmp("proj(q)", s_proj, hs, qw)
k_e = F.linear(hs, kw); v_e = F.linear(hs, vw)
q4 = q_e.view(B, S, H, HD).transpose(1, 2)
k4 = k_e.view(B, S, KVH, HD).transpose(1, 2)
v4 = v_e.view(B, S, KVH, HD).transpose(1, 2)
qr_e, qr_c = cmp("rope(q)", s_rope, q4, cos, sin)
kr_e, _ = cmp("rope(k)", s_rope, k4, cos, sin)
kr = kr_e[:, :, None, :, :].expand(B, KVH, G, S, HD).reshape(B, H, S, HD)
vr = v4[:, :, None, :, :].expand(B, KVH, G, S, HD).reshape(B, H, S, HD)
cmp("attention(q,k,v)", s_attn, qr_e, kr, vr, S)
o_in = torch.randn(B, S, H * HD, device="cuda:0")
cmp("out_proj", s_proj, o_in, ow)

print()
print("--- rope: is the difference an FMA contraction? ---")
cos_e = cos.unsqueeze(1); sin_e = sin.unsqueeze(1)
x = q4
x1 = x[..., : HD // 2]; x2 = x[..., HD // 2:]
rot = torch.cat((-x2, x1), dim=-1)
a = x * cos_e
b = rot * sin_e
unfused = a + b                        # two roundings, as eager does
fma = torch.addcmul(a, rot, sin_e)     # still not a true fma
# true fma via float64 of the *same* two products, rounded once:
exact = (x.double() * cos_e.double() + rot.double() * sin_e.double())
fma_ref = (a.double() + rot.double() * sin_e.double())  # a exact, one product exact, single round
print("compiled_vs_unfused      max_abs =",
      f"{(qr_c.float()-unfused.float()).abs().max().item():.4e}")
print("compiled_vs_fma_model    max_abs =",
      f"{(qr_c.double()-fma_ref).abs().max().item():.4e}")
print("eager_vs_fma_model       max_abs =",
      f"{(unfused.double()-fma_ref).abs().max().item():.4e}")
print("compiled_vs_fp64_exact   max_abs =",
      f"{(qr_c.double()-exact).abs().max().item():.4e}")
print("eager_vs_fp64_exact      max_abs =",
      f"{(unfused.double()-exact).abs().max().item():.4e}")
nc = int((qr_c.double() - fma_ref != 0).sum().item())
print(f"elements where compiled != single-rounding-fma model: {nc} / {qr_c.numel()}")
