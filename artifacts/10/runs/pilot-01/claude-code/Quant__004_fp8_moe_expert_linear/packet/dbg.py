import enum, torch, triton, importlib
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self): return self.value
    enum.StrEnum = StrEnum
import reference as ref
import kernel as K

dev = torch.device("cuda:0")
torch.manual_seed(0)
M = 384
inp = ref.get_inputs({"num_tokens": M}, dev)
hs_, rw_, gu_, dn_ = inp["hidden_states"], inp["routing_weight"], inp["gate_up_weight"], inp["down_weight"]
H = 3584; I = 2048; hk = H // 128; ik = I // 128

# ---- reference intermediates
asc = ref.BlockwiseScaler(ref.ScalingType.BlockWise1x128)
wsc = ref.BlockwiseScaler(ref.ScalingType.BlockWise128x128)
hf = hs_.to(torch.float32)
s_h = asc.compute_scales(hf)
h_q = asc.apply_scaling(hf, s_h, inverse=False, clamp_to_fp8_range=True).to(torch.float8_e4m3fn)
guT = gu_.to(torch.float32).T
s_gu = wsc.compute_scales(guT)
gu_q = wsc.apply_scaling(guT, s_gu, inverse=False, clamp_to_fp8_range=True).T.to(torch.float8_e4m3fn)

# ---- mine
hq = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=dev)
hsq = torch.empty((M, hk), dtype=torch.float32, device=dev)
K._quant_act_kernel[(triton.cdiv(M, 32), hk)](hs_, hq, hsq, M, hs_.stride(0), hq.stride(0), hsq.stride(0), BLOCK_M=32, num_warps=4, num_stages=2)
print("act scale max diff:", (hsq - s_h).abs().max().item())
print("act q bitwise equal:", torch.equal(hq.view(torch.uint8), h_q.view(torch.uint8)),
      " maxdiff:", (hq.float() - h_q.float()).abs().max().item())

NGU = 2 * I
guq = torch.empty((NGU, H), dtype=torch.float8_e4m3fn, device=dev)
gus = torch.empty((NGU // 128, hk), dtype=torch.float32, device=dev)
dnq = torch.empty((H, I), dtype=torch.float8_e4m3fn, device=dev)
dns = torch.empty((H // 128, ik), dtype=torch.float32, device=dev)
nt0 = (NGU // 128) * hk; nt1 = (H // 128) * ik
K._quant_w2_kernel[(nt0 + nt1,)](gu_, guq, gus, gu_.stride(0), guq.stride(0), gus.stride(0),
                                 dn_, dnq, dns, dn_.stride(0), dnq.stride(0), dns.stride(0),
                                 NK0=hk, NTILES0=nt0, NK1=ik, num_warps=8, num_stages=2)
print("w scale max diff:", (gus - s_gu.T).abs().max().item())
print("w q bitwise equal:", torch.equal(guq.view(torch.uint8), gu_q.view(torch.uint8)),
      " maxdiff:", (guq.float() - gu_q.float()).abs().max().item())

# ---- reference gemm1
gemm = ref.CuBLASRefBlockwiseGemm()
gu_out = gemm.scaled_mm(h_q, gu_q, s_h, ref.ScalingType.BlockWise1x128,
                        s_gu.T.contiguous(), ref.ScalingType.BlockWise128x128,
                        None, torch.bfloat16, True)
gate, up = gu_out.chunk(2, dim=-1)
gated = torch.nn.functional.silu(gate) * up

# ---- my gemm1 (using reference-quantized inputs to isolate)
nt_gu = I // 128
gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
gs = torch.empty((M, ik), dtype=torch.float32, device=dev)
bm = 64
nmt = triton.cdiv(M, bm)
K._gemm1_kernel[(nmt * nt_gu,)](hq, hsq, guq, gus, gq, gs, M,
    hq.stride(0), hsq.stride(0), guq.stride(0), gus.stride(0), gq.stride(0), gs.stride(0),
    KBLK=hk, NTILE=nt_gu, IHALF=I, BLOCK_M=bm, GROUP_M=8, NUM_MT=nmt, num_warps=8, num_stages=2)

gf = gated.to(torch.float32)
s_g = asc.compute_scales(gf)
g_q = asc.apply_scaling(gf, s_g, inverse=False, clamp_to_fp8_range=True).to(torch.float8_e4m3fn)
print("gated scale reldiff:", ((gs - s_g).abs() / s_g).max().item())
print("gated q maxdiff:", (gq.float() - g_q.float()).abs().max().item())
d = (gq.float() - g_q.float()).abs()
print("gated q mismatch frac:", (d > 0).float().mean().item())

# raw gemm1 accum check: recompute gate/up in torch from my quantized tensors
a32 = asc.apply_scaling(hq.to(torch.float32), hsq, inverse=True)
b32 = wsc.apply_scaling(guq.to(torch.float32).T, gus.T.contiguous(), inverse=True).T
y = (a32 @ b32.T).to(torch.bfloat16)
print("gemm1 ref-vs-ref recompute:", (y.float() - gu_out.float()).abs().max().item())
