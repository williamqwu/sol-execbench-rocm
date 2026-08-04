import enum
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self): return self.value
    enum.StrEnum = StrEnum

import torch, triton
import kernel as K
import reference as R

dev = "cuda:0"
torch.manual_seed(0)
M = 512
hs = torch.randn(M, 1536, dtype=torch.bfloat16, device=dev)
w = torch.randn(4608, 1536, dtype=torch.bfloat16, device=dev) * 0.05
b = torch.randn(4608, dtype=torch.bfloat16, device=dev)

# ---- reference intermediates ----
asc = R.BlockwiseScaler(R.ScalingType.BlockWise1x128)
wsc = R.BlockwiseScaler(R.ScalingType.BlockWise128x128)
xf = hs.to(torch.float32)
sx_ref = asc.compute_scales(xf)
xs = asc.apply_scaling(xf, sx_ref, inverse=False, clamp_to_fp8_range=True)
qx_ref = xs.to(torch.float8_e4m3fn)

wf = w.T.to(torch.float32)
sw_ref = wsc.compute_scales(wf)
ws = wsc.apply_scaling(wf, sw_ref, inverse=False, clamp_to_fp8_range=True)
qw_ref = ws.T.to(torch.float8_e4m3fn)
swc_ref = sw_ref.T.contiguous()

# ---- my intermediates ----
num_kb = 12
qx = torch.empty((M, 1536), dtype=torch.float8_e4m3fn, device=dev)
sx = torch.empty((M, num_kb), dtype=torch.float32, device=dev)
qw = torch.empty((4608, 1536), dtype=torch.float8_e4m3fn, device=dev)
sw = torch.empty((36, num_kb), dtype=torch.float32, device=dev)
K._quant_act[(triton.cdiv(M, 32), num_kb)](hs, qx, sx, M, hs.stride(0), qx.stride(0),
                                            sx.stride(0), BLOCK_M=32, num_warps=4, num_stages=1)
K._quant_w[(36, num_kb)](w, qw, sw, w.stride(0), qw.stride(0), sw.stride(0),
                         num_warps=8, num_stages=1)

print("sx  exact:", torch.equal(sx, sx_ref), " maxreldiff",
      ((sx - sx_ref).abs() / sx_ref).max().item())
print("sw  exact:", torch.equal(sw, swc_ref), " maxreldiff",
      ((sw - swc_ref).abs() / swc_ref).max().item())

dqx = (qx.float() != qx_ref.float())
dqw = (qw.float() != qw_ref.float())
print("qx mismatch frac:", dqx.float().mean().item())
print("qw mismatch frac:", dqw.float().mean().item())
if dqx.any():
    i = dqx.nonzero()[0]
    r, c = i[0].item(), i[1].item()
    print("  example qx: exact=", xs[r, c].item(), " ref=", qx_ref[r, c].float().item(),
          " mine=", qx[r, c].float().item())
if dqw.any():
    i = dqw.nonzero()[0]
    r, c = i[0].item(), i[1].item()
    print("  example qw: exact=", ws.T[r, c].item(), " ref=", qw_ref[r, c].float().item(),
          " mine=", qw[r, c].float().item())

# ---- GEMM only, fed the REFERENCE quantised tensors ----
out = torch.empty((3, M, 16, 96), dtype=torch.bfloat16, device=dev)
cfg = K._cfg(M)
BM, BN = cfg["BLOCK_M"], cfg["BLOCK_N"]
K._gemm[(triton.cdiv(M, BM) * (4608 // BN),)](
    qx_ref, qw_ref, sx_ref, swc_ref, b, out, M,
    qx_ref.stride(0), qw_ref.stride(0), sx_ref.stride(0), swc_ref.stride(0), 1536,
    NB_PER_OUT=1536 // BN, BLOCK_M=BM, BLOCK_N=BN, GROUP_M=cfg["GROUP_M"],
    NUM_KB=num_kb, num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])

gref = R.CuBLASRefBlockwiseGemm()
y = gref.scaled_mm(qx_ref, qw_ref, sx_ref, R.ScalingType.BlockWise1x128,
                   swc_ref, R.ScalingType.BlockWise128x128, b, torch.bfloat16, True)
y = y.view(M, 3, 16, 96)
tol_a, tol_r = 0.011007797401390344, 0.0078125
for i in range(3):
    d = (out[i].float() - y[:, i].float()).abs()
    th = tol_a + tol_r * y[:, i].float().abs()
    print(f"GEMM-only[{i}] match={(d <= th).float().mean().item():.6f} maxdiff={d.max().item():.5f}")
