#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, "/work/scripts/runners"); sys.path.insert(0, "/work/src")
import torch, torch.nn.functional as F
from _common import exec_reference, load_problem, prepare_inputs
PROB = Path("/work/data/SOL-ExecBench/benchmark/L1/067_flash_attention_gqa_ultralong")
UUID = "b0c05812-9ac0-5ecb-a7a9-73edaf552dde"
H, KVH, HD = 32, 8, 128
definition, workloads = load_problem(PROB)
w = [x for x in workloads if x.uuid == UUID][0]
_, ns = exec_reference(definition)
torch.manual_seed(0)
hs, cos, sin, qw, kw, vw, ow = prepare_inputs(definition, w, ns, device="cuda:0")
B, S, _ = hs.shape
q4 = F.linear(hs, qw).view(B, S, H, HD).transpose(1, 2)

def s_rope(x, cos, sin):
    ce = cos.unsqueeze(1); se = sin.unsqueeze(1)
    x1 = x[..., :HD//2]; x2 = x[..., HD//2:]
    return (x * ce) + (torch.cat((-x2, x1), dim=-1) * se)

e = s_rope(q4, cos, sin)
c = torch.compile(s_rope, dynamic=False)(q4, cos, sin)
torch.cuda.synchronize()

ce = cos.unsqueeze(1).expand(B, H, S, HD).contiguous()
se = sin.unsqueeze(1).expand(B, H, S, HD).contiguous()
rot = torch.cat((-q4[..., HD//2:], q4[..., :HD//2]), dim=-1)
a32 = q4 * ce; b32 = rot * se
xd, cd, rd, sd = q4.double(), ce.double(), rot.double(), se.double()
m_fma2 = (a32.double() + rd*sd).float()
m_fma1 = (xd*cd + b32.double()).float()
m_full = (xd*cd + rd*sd).float()

diff = (c != e)
nd = int(diff.sum().item())
print(f"elements where compiled != eager: {nd}/{c.numel()} = {nd/c.numel():.4f}")
for name, m in [("fma(rot,sin,a32)", m_fma2), ("fma(x,cos,b32)", m_fma1),
                ("full fp64 then round", m_full)]:
    hit = int(((c == m) & diff).sum().item())
    print(f"  of those, compiled bit-equals {name:22s}: {hit:8d} ({hit/nd:.4f})")
    hit_all = int((c == m).sum().item())
    print(f"    overall bit-equal rate: {hit_all/c.numel():.6f}")

# ULP distance on differing elements
ci = c.view(torch.int32).long(); ei = e.view(torch.int32).long()
ulp = (ci - ei).abs()[diff]
print(f"  ULP distance on differing elements: max={ulp.max().item()} "
      f"mean={ulp.double().mean().item():.4f} "
      f"frac==1: {(ulp==1).double().mean().item():.4f}")
# magnitude at max abs diff
d = (c - e).abs()
i = int(d.argmax().item())
print(f"  at max abs diff: eager={e.flatten()[i].item():.9e} "
      f"compiled={c.flatten()[i].item():.9e} "
      f"a32={a32.flatten()[i].item():.9e} b32={b32.flatten()[i].item():.9e} "
      f"exact={m_full.flatten()[i].item():.9e}")
print(f"  |a32|+|b32| at that point = {(a32.abs()+b32.abs()).flatten()[i].item():.6e}"
      f"  ; result magnitude = {abs(e.flatten()[i].item()):.6e}"
      f"  -> cancellation factor {((a32.abs()+b32.abs()).flatten()[i]/e.flatten()[i].abs()).item():.2f}")
