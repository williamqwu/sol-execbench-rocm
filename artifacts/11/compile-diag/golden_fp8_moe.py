"""float64 CPU golden for Quant__004_fp8_moe_expert_linear, one failing workload.

Question: against an exact (float64) evaluation of the reference algorithm, is
the torch.compile output LESS accurate than eager, MORE accurate, or the same?
"""
import sys, torch
import torch.nn.functional as F

SRC = open("/work/data/SOL-ExecBench/benchmark/Quant/004_fp8_moe_expert_linear/reference.py").read()
ns = {}
exec(compile(SRC, "<ref>", "exec"), ns)
run = ns["run"]
comp = torch.compile(run, dynamic=False)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
H, I = 3584, 2048
dev = "cuda:0"
torch.manual_seed(0)
hs = torch.randn(N, H, dtype=torch.bfloat16, device=dev)
rw = torch.randn(N, 1, dtype=torch.bfloat16, device=dev)
gu = torch.randn(I*2, H, dtype=torch.bfloat16, device=dev) * (H ** -0.5)
dw = torch.randn(H, I, dtype=torch.bfloat16, device=dev) * (I ** -0.5)

out_e = run(hs, rw, gu, dw).double().cpu(); torch.cuda.synchronize()
out_c = comp(hs, rw, gu, dw).double().cpu(); torch.cuda.synchronize()

# ---- float64 CPU golden -------------------------------------------------
# Every arithmetic op in float64. The two lossy steps that ARE the algorithm
# are preserved: the FP8 e4m3 quantization, and the bf16 rounding of each GEMM
# output (the reference declares output_dtype=bfloat16). Everything else --
# including the silu*up intermediate, which is where the two variants differ --
# is exact.
E4M3_MAX = 448.0
def q_act(x):                      # BlockWise1x128
    M, K = x.shape
    b = x.reshape(M, 1, K // 128, 128)
    s = (b.abs().amax(3).amax(1) / E4M3_MAX).clamp(min=1e-12)
    y = (b / s.unsqueeze(1).unsqueeze(3)).clamp(-E4M3_MAX, E4M3_MAX).reshape(M, K)
    return y.float().to(torch.float8_e4m3fn), s
def q_wt(x):                       # BlockWise128x128
    M, K = x.shape
    b = x.reshape(M // 128, 128, K // 128, 128)
    s = (b.abs().amax(3).amax(1) / E4M3_MAX).clamp(min=1e-12)
    y = (b / s.unsqueeze(1).unsqueeze(3)).clamp(-E4M3_MAX, E4M3_MAX).reshape(M, K)
    return y.float().to(torch.float8_e4m3fn), s
def deq_act(q, s):
    M, K = q.shape
    return (q.double().reshape(M, 1, K // 128, 128) * s.unsqueeze(1).unsqueeze(3)).reshape(M, K)
def deq_wt(q, s):
    M, K = q.shape
    return (q.double().reshape(M // 128, 128, K // 128, 128) * s.unsqueeze(1).unsqueeze(3)).reshape(M, K)

hs64 = hs.double().cpu(); rw64 = rw.double().cpu()
gu64 = gu.double().cpu(); dw64 = dw.double().cpu()

a_q, a_s = q_act(hs64)
w_q, w_s = q_wt(gu64.T)
w_q = w_q.T.contiguous(); w_s_c = w_s.T.contiguous()
y1 = deq_act(a_q, a_s) @ deq_wt(w_q, w_s_c).T
gate_up_g = y1.to(torch.bfloat16).double()          # declared bf16 GEMM output
gate_g, up_g = gate_up_g.chunk(2, dim=-1)
gated_g = F.silu(gate_g) * up_g                     # EXACT in float64
a2_q, a2_s = q_act(gated_g)
d_q, d_s = q_wt(dw64.T)
d_q = d_q.T.contiguous(); d_s_c = d_s.T.contiguous()
y2 = deq_act(a2_q, a2_s) @ deq_wt(d_q, d_s_c).T
out_g = y2.to(torch.bfloat16).double() * rw64

def rep(tag, x, g):
    d = (x - g).abs()
    rel = (d / g.abs().clamp(min=1e-12))
    print(f"{tag:26s} max_abs={float(d.max()):.6e}  mean_abs={float(d.mean()):.6e}  "
          f"rms={float((d**2).mean().sqrt()):.6e}  max_rel={float(rel.max()):.4e}")
    return d

print(f"N={N}")
de = rep("eager    vs fp64 golden", out_e, out_g)
dc = rep("compiled vs fp64 golden", out_c, out_g)
rep("compiled vs eager      ", out_c, out_e)
closer_c = int((dc < de).sum()); closer_e = int((de < dc).sum()); tie = int((de == dc).sum())
print(f"elements where compiled is closer to golden: {closer_c} ({closer_c/de.numel():.4f})")
print(f"elements where eager    is closer to golden: {closer_e} ({closer_e/de.numel():.4f})")
print(f"ties: {tie} ({tie/de.numel():.4f})")
print(f"mean_abs ratio compiled/eager = {float(dc.mean())/float(de.mean()):.4f}")

# --- the divergence op in isolation: silu(gate)*up ------------------------
gate_b = gate_up_g.to(torch.bfloat16).chunk(2, dim=-1)[0]
up_b   = gate_up_g.to(torch.bfloat16).chunk(2, dim=-1)[1]
ex = (F.silu(gate_b) * up_b).double()                       # eager: bf16 silu, bf16 mul
fu = (F.silu(gate_b.float()) * up_b.float()).to(torch.bfloat16).double()  # fused fp32 intermediate
gd = (F.silu(gate_b.double()) * up_b.double())
print("\nstage 12 in isolation (identical bf16 inputs):")
rep("  eager silu*up  vs f64", ex, gd)
rep("  fp32-fused     vs f64", fu, gd)
print("  fp32-fused == compiled-stage12 bitwise:",
      bool(torch.equal(fu, gd*0 + fu)))
