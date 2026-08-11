"""Is the FP8 requantizer the amplifier?

Take the two stage-12 tensors (eager's and Inductor's, which differ by <=1 bf16
ULP) and push both through the SAME down-projection two ways:
  (A) the reference's FP8 path (quantize -> dequantize -> fp32 matmul)
  (B) a plain fp32 matmul with the same dequantized weights, no activation
      quantization at all
and compare how far apart the two outputs end up in each case.
"""
import torch, torch.nn.functional as F
SRC = open("/work/data/SOL-ExecBench/benchmark/Quant/004_fp8_moe_expert_linear/reference.py").read()
ns = {}; exec(compile(SRC, "<ref>", "exec"), ns)
BlockwiseScaler, ScalingType = ns["BlockwiseScaler"], ns["ScalingType"]

N, H, I = 1024, 3584, 2048
dev = "cuda:0"
torch.manual_seed(0)
hs = torch.randn(N, H, dtype=torch.bfloat16, device=dev)
rw = torch.randn(N, 1, dtype=torch.bfloat16, device=dev)
gu = torch.randn(I*2, H, dtype=torch.bfloat16, device=dev) * (H ** -0.5)
dw = torch.randn(H, I, dtype=torch.bfloat16, device=dev) * (I ** -0.5)

# stage 11 (bit-identical between eager and compiled -- measured), then stage 12
def upto11(hidden_states, gate_up_weight):
    a = BlockwiseScaler(ScalingType.BlockWise1x128); w = BlockwiseScaler(ScalingType.BlockWise128x128)
    hf = hidden_states.to(torch.float32); sh = a.compute_scales(hf)
    gt = gate_up_weight.to(torch.float32).T; sg = w.compute_scales(gt)
    hq = a.apply_scaling(hf, sh, False, True).to(torch.float8_e4m3fn)
    gq = w.apply_scaling(gt, sg, False, True).T.to(torch.float8_e4m3fn)
    sgc = sg.T.contiguous()
    y = a.apply_scaling(hq.to(torch.float32), sh, inverse=True) @ \
        w.apply_scaling(gq.to(torch.float32), sgc, inverse=True).T
    return y.to(torch.bfloat16)

gate_up = upto11(hs, gu)
gate, up = gate_up.chunk(2, dim=-1)
g_eager = F.silu(gate) * up                                      # eager stage 12
g_fused = (F.silu(gate.float()) * up.float()).to(torch.bfloat16) # Inductor stage 12
d12 = (g_eager.double() - g_fused.double()).abs()
print(f"stage 12 gap: max_abs={float(d12.max()):.6e} mean_abs={float(d12.mean()):.6e} "
      f"frac_diff={float((d12>0).float().mean()):.6f}")

w = BlockwiseScaler(ScalingType.BlockWise128x128); a = BlockwiseScaler(ScalingType.BlockWise1x128)
dt = dw.to(torch.float32).T; sd = w.compute_scales(dt)
dq = w.apply_scaling(dt, sd, False, True).T.to(torch.float8_e4m3fn)
sdc = sd.T.contiguous()
B = w.apply_scaling(dq.to(torch.float32), sdc, inverse=True)    # dequantized weight, shared

def pathA(g):    # reference FP8 activation path
    gf = g.to(torch.float32); s = a.compute_scales(gf)
    q = a.apply_scaling(gf, s, False, True).to(torch.float8_e4m3fn)
    return ((a.apply_scaling(q.to(torch.float32), s, inverse=True) @ B.T).to(torch.bfloat16) * rw).double()
def pathB(g):    # same GEMM, no activation quantization
    return ((g.to(torch.float32) @ B.T).to(torch.bfloat16) * rw).double()

for tag, p in (("A  FP8-quantized activations", pathA), ("B  no activation quantization", pathB)):
    d = (p(g_eager) - p(g_fused)).abs()
    print(f"{tag:30s}: out gap max_abs={float(d.max()):.6e} mean_abs={float(d.mean()):.6e} "
          f"frac>atol(4.752e-3)={float((d>4.752364921349595e-3).float().mean()):.6f}")

# how many FP8 codes flip
def codes(g):
    gf = g.to(torch.float32); s = a.compute_scales(gf)
    return a.apply_scaling(gf, s, False, True).to(torch.float8_e4m3fn).view(torch.uint8)
ce, cf = codes(g_eager), codes(g_fused)
nd = int((ce != cf).sum())
print(f"FP8 activation codes differing: {nd} / {ce.numel()} = {nd/ce.numel():.6f}")
