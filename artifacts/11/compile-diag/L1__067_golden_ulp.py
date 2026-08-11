#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, "/work/scripts/runners"); sys.path.insert(0, "/work/src")
import torch, torch.nn.functional as F
from _common import exec_reference, load_problem, prepare_inputs
PROB = Path("/work/data/SOL-ExecBench/benchmark/L1/067_flash_attention_gqa_ultralong")
UUID = sys.argv[1]
H, KVH, HD, G = 32, 8, 128, 4
SC = HD ** -0.5
definition, workloads = load_problem(PROB)
w = [x for x in workloads if x.uuid == UUID][0]
ref_run, ns = exec_reference(definition)
torch.manual_seed(0)
ins = prepare_inputs(definition, w, ns, device="cuda:0")

def golden(t):
    hs, cos, sin, qw, kw, vw, ow = t
    B, S, _ = hs.shape
    q = F.linear(hs, qw).view(B,S,H,HD).transpose(1,2)
    k = F.linear(hs, kw).view(B,S,KVH,HD).transpose(1,2)
    v = F.linear(hs, vw).view(B,S,KVH,HD).transpose(1,2)
    ce, se = cos.unsqueeze(1), sin.unsqueeze(1)
    def rope(x):
        return (x*ce) + (torch.cat((-x[...,HD//2:], x[...,:HD//2]), -1)*se)
    q, k = rope(q), rope(k)
    k = k[:,:,None].expand(B,KVH,G,S,HD).reshape(B,H,S,HD)
    v = v[:,:,None].expand(B,KVH,G,S,HD).reshape(B,H,S,HD)
    aw = torch.matmul(q, k.transpose(2,3))*SC
    aw = aw + torch.triu(torch.full((S,S), float('-inf'), dtype=hs.dtype), 1)
    aw = F.softmax(aw, -1, dtype=torch.float64)
    o = torch.matmul(aw, v).transpose(1,2).contiguous().reshape(B,S,H*HD)
    return F.linear(o, ow)

g = golden(tuple(t.detach().cpu().double() for t in ins))
oe = ref_run(*ins).detach().cpu().double()
ns2 = {}; exec(compile(definition.reference, "<r>", "exec"), ns2)
oc = torch.compile(ns2["run"], dynamic=False)(*ins).detach().cpu().double()

de = (oe-g).abs(); dc = (oc-g).abs()
i = int(de.argmax()); j = int(dc.argmax())
print(f"uuid={UUID} axes={dict(w.axes)} out.shape={tuple(g.shape)}")
print(f"eager argmax idx={i} golden={g.flatten()[i].item():.9f} "
      f"eager={oe.flatten()[i].item():.9f} compiled={oc.flatten()[i].item():.9f} "
      f"err_e={de.flatten()[i].item():.6e} err_c={dc.flatten()[i].item():.6e}")
print(f"compiled argmax idx={j} same_idx={i==j}")
val = abs(g.flatten()[i].item())
import math
ulp = 2.0**(math.floor(math.log2(val)) - 23)
print(f"|value|={val:.6f}  ulp(fp32)={ulp:.6e}  err/ulp = {de.flatten()[i].item()/ulp:.4f}")
print(f"eager==compiled bitwise at that element: "
      f"{oe.flatten()[i].item()==oc.flatten()[i].item()}")
print(f"output abs max = {g.abs().max().item():.4f}, "
      f"mean|out| = {g.abs().mean().item():.4f}")
# distribution of error/ulp
uv = torch.pow(2.0, torch.floor(torch.log2(g.abs().clamp_min(1e-30))) - 23)
print(f"eager    err/ulp: p50={(de/uv).median().item():.3f} "
      f"p99={(de/uv).flatten().kthvalue(int(0.99*de.numel()))[0].item():.3f} "
      f"max={(de/uv).max().item():.3f}")
print(f"compiled err/ulp: p50={(dc/uv).median().item():.3f} "
      f"p99={(dc/uv).flatten().kthvalue(int(0.99*dc.numel()))[0].item():.3f} "
      f"max={(dc/uv).max().item():.3f}")
