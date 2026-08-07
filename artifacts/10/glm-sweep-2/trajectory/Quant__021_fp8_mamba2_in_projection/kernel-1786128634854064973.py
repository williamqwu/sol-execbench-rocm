import torch
import torch.nn.functional as F

E4M3_MAX = 448.0


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    M, K = hidden_states.shape
    N, K_w = weight.shape

    # --- Compute blockwise scales (match reference exactly) ---
    # Activation: BlockWise1x128 -> scale_x (M, K//128)
    x_fp32 = hidden_states.to(torch.float32)
    x_blk = x_fp32.reshape(M, K // 128, 128)
    scale_x = torch.clamp(x_blk.abs().amax(dim=2) / E4M3_MAX, min=1e-12)  # (M, K//128)

    # Weight: BlockWise128x128 -> scale_w (N//128, K//128)
    w_fp32 = weight.to(torch.float32)
    w_blk = w_fp32.reshape(N // 128, 128, K // 128, 128)
    scale_w = torch.clamp(w_blk.abs().amax(dim=3).amax(dim=1) / E4M3_MAX, min=1e-12)  # (N//128, K//128)

    # --- Quantize to fp8 then dequantize to bf16 (fused) ---
    # x: (M, K//128, 128) / scale (M, K//128, 1) -> clamp -> fp8 -> * scale -> bf16
    sx = scale_x.unsqueeze(2)  # (M, K//128, 1)
    qx = torch.clamp(x_blk / sx, -E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    a_bf16 = (qx.to(torch.float32) * sx).reshape(M, K).to(torch.bfloat16)

    # w: (N//128, 128, K//128, 128) / scale (N//128, 1, K//128, 1) -> clamp -> fp8 -> * scale -> bf16
    sw = scale_w.unsqueeze(1).unsqueeze(3)  # (N//128, 1, K//128, 1)
    qw = torch.clamp(w_blk / sw, -E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    b_bf16 = (qw.to(torch.float32) * sw).reshape(N, K).to(torch.bfloat16)

    # bf16 matmul (MFMA accumulates in fp32)
    output = a_bf16 @ b_bf16.T
    return output
