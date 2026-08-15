import torch
import torch.nn.functional as F


def fp8_linear_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 linear layer matching reference implementation exactly.
    Optimized for minimal memory traffic.
    """
    M, K = x.shape
    N = weight.shape[0]

    # Convert to FP32
    x_fp32 = x.float()
    weight_fp32 = weight.float()

    # === Activation scaling (BlockWise1x128) ===
    x_reshaped = x_fp32.reshape(M, K // 128, 128)
    x_block_max = x_reshaped.abs().amax(dim=2)  # (M, K//128)
    scale_x = x_block_max.clamp(min=1e-12) / 448.0

    # Apply scaling and quantize
    scale_x_expanded = scale_x.unsqueeze(2)  # (M, K//128, 1)
    x_scaled = (x_reshaped / scale_x_expanded).clamp(-448.0, 448.0)
    qx = x_scaled.reshape(M, K).to(torch.float8_e4m3fn)

    # === Weight scaling (BlockWise128x128) ===
    weight_t = weight_fp32.T  # (K, N)
    weight_reshaped = weight_t.reshape(K // 128, 128, N // 128, 128)
    weight_block_max = weight_reshaped.abs().amax(dim=3).amax(dim=1)  # (K//128, N//128)
    scale_w_t = weight_block_max.clamp(min=1e-12) / 448.0

    # Apply scaling and quantize
    scale_w_expanded = scale_w_t.unsqueeze(1).unsqueeze(3)  # (K//128, 1, N//128, 1)
    weight_scaled = (weight_reshaped / scale_w_expanded).clamp(-448.0, 448.0)
    qw = weight_scaled.reshape(K, N).T.to(torch.float8_e4m3fn)  # (N, K)

    # === FP8 GEMM (dequantize and matmul) ===
    # Dequantize activations
    qx_f32 = qx.float()
    qx_reshaped = qx_f32.reshape(M, K // 128, 128)
    x_dequant = (qx_reshaped * scale_x_expanded).reshape(M, K)

    # Dequantize weights
    qw_t = qw.float().T  # (K, N)
    qw_reshaped = qw_t.reshape(K // 128, 128, N // 128, 128)
    w_dequant = (qw_reshaped * scale_w_expanded).reshape(K, N)

    # Matmul in FP32
    output = x_dequant @ w_dequant

    return output.to(torch.bfloat16)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 Shared Expert MLP with SwiGLU activation.

    Architecture: x -> [gate_proj, up_proj] -> silu(gate) * up -> down_proj -> output
    """
    # FP8 gate and up projections
    gate = fp8_linear_fused(hidden_states, gate_proj_weight)
    up = fp8_linear_fused(hidden_states, up_proj_weight)

    # SwiGLU activation in BF16
    intermediate = F.silu(gate) * up

    # FP8 down projection
    output = fp8_linear_fused(intermediate, down_proj_weight)

    return output
