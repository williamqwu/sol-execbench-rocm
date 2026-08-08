import sys, torch
sys.path.insert(0, "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear")
import reference as R
import tk
import torch.nn.functional as F

dev = torch.device("cuda")
torch.manual_seed(0)
nt = 512
inp = R.get_inputs({"num_tokens": nt}, dev)
hs, rw, guw, dw = inp["hidden_states"], inp["routing_weight"], inp["gate_up_weight"], inp["down_weight"]

asc_ref = R.BlockwiseScaler(R.ScalingType.BlockWise1x128)
wsc_ref = R.BlockwiseScaler(R.ScalingType.BlockWise128x128)

h32 = hs.float()
s_h = asc_ref.compute_scales(h32)
h_fp8 = asc_ref.apply_scaling(h32, s_h, False, True).to(torch.float8_e4m3fn)

gw_t = guw.float().T
s_gu = wsc_ref.compute_scales(gw_t)
gu_fp8 = wsc_ref.apply_scaling(gw_t, s_gu, False, True).T.to(torch.float8_e4m3fn)

aq, asc = tk.quant_act(hs)
wq, wsc = tk.quant_weight(guw)
print("act q match:", (aq.view(torch.uint8) == h_fp8.view(torch.uint8)).float().mean().item())
print("act s match:", (asc - s_h).abs().max().item())
print("w q match:", (wq.view(torch.uint8) == gu_fp8.view(torch.uint8)).float().mean().item())
print("w s match:", (wsc - s_gu.T).abs().max().item())

# reference gemm1
gemm = R.CuBLASRefBlockwiseGemm()
gu_out = gemm.scaled_mm(h_fp8, gu_fp8, s_h, R.ScalingType.BlockWise1x128,
                        s_gu.T.contiguous(), R.ScalingType.BlockWise128x128,
                        None, torch.bfloat16, True)
gate, up = gu_out.chunk(2, dim=-1)
gated = F.silu(gate) * up
s_g = asc_ref.compute_scales(gated.float())
g_fp8 = asc_ref.apply_scaling(gated.float(), s_g, False, True).to(torch.float8_e4m3fn)

# my gemm1
import triton
M, H = hs.shape
I = 2048
gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
gs = torch.empty((M, I // 128), dtype=torch.float32, device=dev)
tk._gemm1_silu_quant[(triton.cdiv(M, 128), I // 128)](
    aq, asc, wq, wsc, gq, gs, M, H, I,
    aq.stride(0), asc.stride(0), wq.stride(0), wsc.stride(0),
    gq.stride(0), gs.stride(0), BLOCK_M=128, NUM_K=H // 128,
    num_warps=8, num_stages=2)
print("gated s rel:", ((gs - s_g).abs() / s_g).max().item())
d = (gq.float() * gs.repeat_interleave(128, 1)) - (g_fp8.float() * s_g.repeat_interleave(128, 1))
print("gated deq maxabs:", d.abs().max().item(), "ref scale", gated.abs().max().item())
print("gq exact frac:", (gq.view(torch.uint8) == g_fp8.view(torch.uint8)).float().mean().item())

