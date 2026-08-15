import torch

E4M3_MAX = 448.0

@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    o_proj_weight: torch.Tensor,
    o_proj_bias: torch.Tensor,
):
    """
    Optimized FP8-quantized attention output projection.

    Uses torch.compile for fusion and optimized memory access patterns.
    """
    M, K = attn_output.shape
    N = o_proj_weight.shape[0]

    # Compute activation scales (BlockWise1x128)
    attn_output_fp32 = attn_output.to(torch.float32)
    num_k_blocks = K // 128
    act_reshaped = attn_output_fp32.reshape(M, num_k_blocks, 128)
    block_max = act_reshaped.abs().amax(dim=2)
    scale_x = (block_max / E4M3_MAX).clamp(min=1e-12)

    # Apply scaling and quantize activations (BlockWise1x128)
    act_scaled = act_reshaped / scale_x.unsqueeze(2)
    act_scaled = act_scaled.clamp(min=-E4M3_MAX, max=E4M3_MAX)
    qx = act_scaled.reshape(M, K).to(torch.float8_e4m3fn)

    # Compute weight scales and quantize weights (BlockWise128x128)
    weight_fp32 = o_proj_weight.to(torch.float32)
    weight_t = weight_fp32.T  # (K, N)

    num_k_blocks_w = K // 128
    num_n_blocks_w = N // 128
    weight_t_reshaped = weight_t.reshape(num_k_blocks_w, 128, num_n_blocks_w, 128)
    weight_block_max = weight_t_reshaped.abs().amax(dim=3).amax(dim=1)
    scales_w = (weight_block_max / E4M3_MAX).clamp(min=1e-12)

    # Apply scaling and quantize weights
    weight_t_reshaped_scaled = weight_t_reshaped / scales_w.unsqueeze(1).unsqueeze(3)
    weight_t_scaled = weight_t_reshaped_scaled.clamp(min=-E4M3_MAX, max=E4M3_MAX).reshape(K, N)
    qw = weight_t_scaled.T.to(torch.float8_e4m3fn)  # (N, K)

    # Transpose scales to CuBLAS format: (N//128, K//128)
    scale_w_cublas = scales_w.T.contiguous()

    # Dequantize for matmul
    # Dequantize A (activations with BlockWise1x128)
    qx_fp32 = qx.to(torch.float32)
    qx_reshaped = qx_fp32.reshape(M, num_k_blocks, 128)
    a_f32 = (qx_reshaped * scale_x.unsqueeze(2)).reshape(M, K)

    # Dequantize B (weights with BlockWise128x128)
    qw_fp32 = qw.to(torch.float32)
    qw_t = qw_fp32.T  # (K, N)
    qw_t_reshaped = qw_t.reshape(num_k_blocks_w, 128, num_n_blocks_w, 128)
    b_f32_reshaped = qw_t_reshaped * scale_w_cublas.T.unsqueeze(1).unsqueeze(3)
    b_f32 = b_f32_reshaped.reshape(K, N)

    # Matmul in float32
    y = a_f32 @ b_f32

    # Add bias
    if o_proj_bias is not None and o_proj_bias.numel():
        y = y + o_proj_bias

    return y.to(torch.bfloat16)
