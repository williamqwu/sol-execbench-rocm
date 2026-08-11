#!/usr/bin/env python3
"""ADVERSARIAL CHECK 3: is the divergence really generated at the RoPE
pointwise (reference.py:41/47), or somewhere else (e.g. a GEMM algorithm swap)?

Three independent angles, all on uuid b0c05812 (bs=2, seq=128):

 (1) UNPERTURBED ABLATION. Run the *unmodified* compiled graph, but feed
     cos = 1, sin = 0. The graph, the kernels and the fusion are byte-identical
     -- only the runtime values change. With sin = 0 the RoPE line degenerates
     to x*1 + rot*0 = x, which any association or FMA contraction computes
     exactly, so a contraction there can no longer inject error. Whatever
     remains is generated downstream. This is the one test that does not
     change what Inductor compiles.

 (2) PER-STAGE ISOLATION. Compile each stage alone, feed it eager-produced
     inputs that are bitwise identical, and see which stages generate error.

 (3) CODEGEN. Grep the Inductor output for extern_kernels.bmm /
     scaled_dot_product to test the "it swapped in SDPA" hypothesis.
"""
from __future__ import annotations
import sys, glob, os, json
from pathlib import Path

sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")
import torch
import torch.nn.functional as F
from _common import exec_reference, load_problem, prepare_inputs

PROB = Path("/work/data/SOL-ExecBench/benchmark/L1/067_flash_attention_gqa_ultralong")
UU = "b0c05812"
H, KVH, D, G = 32, 8, 128, 4
SC = D ** -0.5

definition, workloads = load_problem(PROB)
ref_run, ref_ns = exec_reference(definition)
w = [x for x in workloads if x.uuid.startswith(UU)][0]
atol = float(w.tolerance.max_atol); rtol = float(w.tolerance.max_rtol)
torch.manual_seed(0)
hs, cos, sin, qw, kw, vw, ow = prepare_inputs(definition, w, ref_ns, device="cuda:0")


def d(a, b):
    ae = (a.float() - b.float()).abs()
    return float(ae.max()), float((ae > 0).float().mean())


# ---------- (1) unperturbed ablation ----------
ns2: dict = {}
exec(compile(definition.reference, "<reference>", "exec"), ns2)
full_c = torch.compile(ns2["run"], dynamic=False)

r_real = ref_run(hs, cos, sin, qw, kw, vw, ow)
c_real = full_c(hs, cos, sin, qw, kw, vw, ow)
torch.cuda.synchronize()
print("ABLATION real cos/sin      : max_abs=%.4e frac_diff=%.4f" % d(c_real, r_real), flush=True)

one, zero = torch.ones_like(cos), torch.zeros_like(sin)
r_off = ref_run(hs, one, zero, qw, kw, vw, ow)
c_off = full_c(hs, one, zero, qw, kw, vw, ow)
torch.cuda.synchronize()
print("ABLATION cos=1 sin=0 (RoPE off): max_abs=%.4e frac_diff=%.4f" % d(c_off, r_off), flush=True)

# control: same magnitudes but RoPE live -- cos=sin=1/sqrt2 keeps |q| the same
import math
h = 1.0 / math.sqrt(2.0)
r_h = ref_run(hs, torch.full_like(cos, h), torch.full_like(sin, h), qw, kw, vw, ow)
c_h = full_c(hs, torch.full_like(cos, h), torch.full_like(sin, h), qw, kw, vw, ow)
torch.cuda.synchronize()
print("ABLATION cos=sin=1/sqrt2 (RoPE live, norm preserved): max_abs=%.4e frac_diff=%.4f"
      % d(c_h, r_h), flush=True)

# ---------- (2) per-stage isolation ----------
def rope(x, c, s):
    x1 = x[..., : D // 2]; x2 = x[..., D // 2:]
    rot = torch.cat((-x2, x1), dim=-1)
    return (x * c) + (rot * s)


B, S, _ = hs.shape
q = F.linear(hs, qw).view(B, S, H, D).transpose(1, 2)
k = F.linear(hs, kw).view(B, S, KVH, D).transpose(1, 2)
ce, se = cos.unsqueeze(1), sin.unsqueeze(1)
qr = rope(q, ce, se)
kr = rope(k, ce, se)
kr_e = kr[:, :, None, :, :].expand(B, KVH, G, S, D).reshape(B, H, S, D)
v = F.linear(hs, vw).view(B, S, KVH, D).transpose(1, 2)
v_e = v[:, :, None, :, :].expand(B, KVH, G, S, D).reshape(B, H, S, D)
aw = torch.matmul(qr, kr_e.transpose(2, 3)) * SC
mask = torch.triu(torch.full((S, S), float("-inf"), device=hs.device, dtype=hs.dtype), diagonal=1)
awm = aw + mask
p = F.softmax(awm, dim=-1, dtype=torch.float32).to(q.dtype)
ao = torch.matmul(p, v_e).transpose(1, 2).contiguous().reshape(B, S, H * D)
torch.cuda.synchronize()

cases = [
  ("F.linear q_proj",       lambda x, W: F.linear(x, W),                       (hs, qw)),
  ("rope(q)  [ref.py:41]",  rope,                                              (q, ce, se)),
  ("rope(k)  [ref.py:47]",  rope,                                              (k, ce, se)),
  ("qk matmul * scaling",   lambda a, b: torch.matmul(a, b.transpose(2, 3)) * SC, (qr, kr_e)),
  ("mask + softmax",        lambda a, m: F.softmax(a + m, dim=-1, dtype=torch.float32).to(torch.float32), (aw, mask)),
  ("softmax alone",         lambda a: F.softmax(a, dim=-1, dtype=torch.float32).to(torch.float32), (awm,)),
  ("attn.V matmul",         lambda a, b: torch.matmul(a, b),                   (p, v_e)),
  ("F.linear out_proj",     lambda x, W: F.linear(x, W),                       (ao, ow)),
]
print("\nPER-STAGE ISOLATION (each compiled alone, bitwise-identical eager inputs)", flush=True)
for name, f, args in cases:
    re_ = f(*args)
    ci = torch.compile(f, dynamic=False)(*args)
    torch.cuda.synchronize()
    m, fr = d(ci, re_)
    print(f"  {name:24s} max_abs={m:.4e} frac_diff={fr:.4f}", flush=True)

# ---------- FMA model check on rope(q) ----------
cq = torch.compile(rope, dynamic=False)(q, ce, se)
eq = rope(q, ce, se)
x = q; c_, s_ = ce.expand_as(q), se.expand_as(q)
x1 = x[..., : D // 2]; x2 = x[..., D // 2:]
rot = torch.cat((-x2, x1), dim=-1)
f64 = lambda t: t.double()
m_a = (torch.addcmul(f64(rot * s_), f64(x), f64(c_))).float()      # fma(x,cos, fp32(rot*sin))
m_b = (torch.addcmul(f64(x * c_), f64(rot), f64(s_))).float()      # fma(rot,sin, fp32(x*cos))
m_e = ((f64(x) * f64(c_)) + (f64(rot) * f64(s_))).float()          # no intermediate rounding
diff = cq != eq
n = int(diff.sum())
print(f"\nrope(q): differing elements {n}/{cq.numel()} ({n/cq.numel():.4f})", flush=True)
for nm, mdl in (("fma(x,cos,fp32(rot*sin))", m_a), ("fma(rot,sin,fp32(x*cos))", m_b),
                ("fp64 both, one rounding", m_e)):
    print(f"  compiled bit-equals {nm:26s} on differing elts: "
          f"{float((cq[diff] == mdl[diff]).float().mean()):.4f}   overall: "
          f"{float((cq == mdl).float().mean()):.6f}", flush=True)
print(f"  eager    bit-equals two-roundings model overall: "
      f"{float((eq == ((x*c_) + (rot*s_))).float().mean()):.6f}", flush=True)

# ---------- (3) codegen ----------
cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
if cache:
    hits = {"scaled_dot_product": 0, "extern_kernels.bmm": 0, "extern_kernels.mm": 0,
            "flash_attention": 0, "triton_poi_fused_add_cat_mul": 0}
    files = glob.glob(cache + "/**/output_code.py", recursive=True) + \
            glob.glob(cache + "/**/*.py", recursive=True)
    for fp in set(files):
        try:
            t = Path(fp).read_text()
        except Exception:
            continue
        for kk in hits:
            hits[kk] += t.count(kk)
    print("\nCODEGEN grep over %d cached files: %s" % (len(set(files)), json.dumps(hits)), flush=True)
print("DONE")
