#!/usr/bin/env python3
"""Mechanism tests + float64 CPU golden for L1__067."""
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
ref_run, ref_ns = exec_reference(definition)
torch.manual_seed(0)
ins = prepare_inputs(definition, w, ref_ns, device="cuda:0")
hs, cos, sin, qw, kw, vw, ow = ins
B, S, _ = hs.shape
print(f"### uuid={UUID}  B={B} S={S}")

q4 = F.linear(hs, qw).view(B, S, H, HD).transpose(1, 2)
k4 = F.linear(hs, kw).view(B, S, KVH, HD).transpose(1, 2)
v4 = F.linear(hs, vw).view(B, S, KVH, HD).transpose(1, 2)

# ---------------- (a) FMA contraction test on RoPE ----------------
print("\n=== (a) RoPE: is compiled == an FMA contraction of eager's expression? ===")
def s_rope(x, cos, sin):
    ce = cos.unsqueeze(1); se = sin.unsqueeze(1)
    x1 = x[..., :HD//2]; x2 = x[..., HD//2:]
    rot = torch.cat((-x2, x1), dim=-1)
    return (x * ce) + (rot * se)

eager_rope = s_rope(q4, cos, sin)
comp_rope = torch.compile(s_rope, dynamic=False)(q4, cos, sin)
torch.cuda.synchronize()

ce = cos.unsqueeze(1).expand(B, H, S, HD); se = sin.unsqueeze(1).expand(B, H, S, HD)
x1 = q4[..., :HD//2]; x2 = q4[..., HD//2:]
rot = torch.cat((-x2, x1), dim=-1)
a32 = (q4 * ce)            # fp32 rounded first product
b32 = (rot * se)           # fp32 rounded second product
xd, cd, rd, sd = q4.double(), ce.double(), rot.double(), se.double()
model_eager = (a32 + b32)                            # two roundings (what ATen does)
model_fma2  = (a32.double() + rd * sd).float()       # fma(rot, sin, a)  -> 2nd product unrounded
model_fma1  = (xd * cd + b32.double()).float()       # fma(x, cos, b)    -> 1st product unrounded
n = comp_rope.numel()
for name, m in [("eager (2 roundings)", model_eager),
                ("fma(rot,sin,a32)", model_fma2),
                ("fma(x,cos,b32)", model_fma1)]:
    eq = int((comp_rope == m).sum().item())
    print(f"  compiled bit-equals {name:22s}: {eq:9d}/{n} = {eq/n:.6f}")
eq_e = int((eager_rope == model_eager).sum().item())
print(f"  eager    bit-equals {'eager model':22s}: {eq_e:9d}/{n} = {eq_e/n:.6f}")
print(f"  compiled_vs_eager max_abs = {(comp_rope-eager_rope).abs().max().item():.4e}")

# ---------------- (b) attention sub-stages ----------------
print("\n=== (b) attention sub-stages, identical inputs ===")
qr = eager_rope
kr = s_rope(k4, cos, sin)[:, :, None, :, :].expand(B, KVH, G, S, HD).reshape(B, H, S, HD)
vr = v4[:, :, None, :, :].expand(B, KVH, G, S, HD).reshape(B, H, S, HD)

def s_qk(q, k):
    return torch.matmul(q, k.transpose(2, 3)) * SCALING

def s_softmax(aw, seq_len, dt):
    cm = torch.triu(torch.full((seq_len, seq_len), float('-inf'),
                    device=aw.device, dtype=dt), diagonal=1)
    aw = aw + cm
    return F.softmax(aw, dim=-1, dtype=torch.float32).to(dt)

def s_av(p, v):
    return torch.matmul(p, v)

def rep(name, fn, *args):
    e = fn(*args); c = torch.compile(fn, dynamic=False)(*args)
    torch.cuda.synchronize()
    d = (e.float() - c.float()).abs()
    d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
    nd = int((d > 0).sum().item())
    print(f"  {name:32s} max_abs={d.max().item():.4e} frac_diff={nd/e.numel():.4f}")
    return e

aw_e = rep("qk_matmul * scaling", s_qk, qr, kr)
p_e = rep("mask+softmax", s_softmax, aw_e, S, torch.float32)
rep("av_matmul", s_av, p_e, vr)

# softmax alone without the mask add, to separate them
def s_softmax_only(aw):
    return F.softmax(aw, dim=-1, dtype=torch.float32).to(aw.dtype)
rep("softmax alone (no mask)", s_softmax_only, aw_e)

# qk matmul WITHOUT the trailing scale, to separate bmm from the fused mul
def s_qk_raw(q, k):
    return torch.matmul(q, k.transpose(2, 3))
rep("qk_matmul raw (no scale)", s_qk_raw, qr, kr)

# ---------------- (c) float64 CPU golden ----------------
print("\n=== (c) float64 CPU golden: who is closer, eager or compiled? ===")
def golden(ins64):
    hs, cos, sin, qw, kw, vw, ow = ins64
    B, S, _ = hs.shape
    q = F.linear(hs, qw).view(B, S, H, HD).transpose(1, 2)
    k = F.linear(hs, kw).view(B, S, KVH, HD).transpose(1, 2)
    v = F.linear(hs, vw).view(B, S, KVH, HD).transpose(1, 2)
    ce = cos.unsqueeze(1); se = sin.unsqueeze(1)
    def rope(x):
        x1 = x[..., :HD//2]; x2 = x[..., HD//2:]
        return (x * ce) + (torch.cat((-x2, x1), dim=-1) * se)
    q = rope(q); k = rope(k)
    k = k[:, :, None, :, :].expand(B, KVH, G, S, HD).reshape(B, H, S, HD)
    v = v[:, :, None, :, :].expand(B, KVH, G, S, HD).reshape(B, H, S, HD)
    aw = torch.matmul(q, k.transpose(2, 3)) * SCALING
    cm = torch.triu(torch.full((S, S), float('-inf'), device=hs.device,
                    dtype=hs.dtype), diagonal=1)
    aw = aw + cm
    aw = F.softmax(aw, dim=-1, dtype=torch.float64)
    o = torch.matmul(aw, v).transpose(1, 2).contiguous().reshape(B, S, H*HD)
    return F.linear(o, ow)

ins64 = tuple(t.detach().cpu().double() for t in ins)
g = golden(ins64)
out_eager = ref_run(*ins).detach().cpu().double()
ns2 = {}
exec(compile(definition.reference, "<ref>", "exec"), ns2)
cfn = torch.compile(ns2["run"], dynamic=False)
out_comp = cfn(*ins).detach().cpu().double()
torch.cuda.synchronize()

def err(x, ref, label):
    d = (x - ref).abs()
    rel = d / ref.abs().clamp_min(1e-30)
    print(f"  {label:28s} max_abs={d.max().item():.6e}  "
          f"mean_abs={d.mean().item():.6e}  max_rel={rel.max().item():.6e}  "
          f"rms={d.pow(2).mean().sqrt().item():.6e}")
    return d.max().item(), d.pow(2).mean().sqrt().item()

ee = err(out_eager, g, "eager   vs fp64 golden")
ec = err(out_comp, g, "compiled vs fp64 golden")
err(out_comp, out_eager, "compiled vs eager")
print(f"  -> compiled RMS / eager RMS = {ec[1]/ee[1]:.4f}   "
      f"(<1 means compiled is MORE accurate)")
print(f"  -> compiled maxabs / eager maxabs = {ec[0]/ee[0]:.4f}")
closer = ((out_comp - g).abs() < (out_eager - g).abs()).float().mean().item()
tie = ((out_comp - g).abs() == (out_eager - g).abs()).float().mean().item()
print(f"  -> fraction of elements where compiled is strictly closer to golden: {closer:.4f}"
      f"  (tie {tie:.4f}, eager closer {1-closer-tie:.4f})")

tol = w.tolerance
atol = float(tol.max_atol); rtol = float(tol.max_rtol)
for label, x in [("eager", out_eager), ("compiled", out_comp)]:
    bad = ((x - g).abs() > (atol + rtol * g.abs()))
    print(f"  {label:9s} matched_ratio against fp64 GOLDEN under workload tol "
          f"(atol={atol:.3e}): {1-bad.float().mean().item():.6f}")
