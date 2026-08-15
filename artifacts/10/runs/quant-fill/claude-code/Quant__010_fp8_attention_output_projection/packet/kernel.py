import torch


def compute_blockwise_scales(tensor, block_m, block_k):
    """Compute blockwise scales matching the reference implementation."""
    M, K = tensor.shape
    E4M3_MAX = 448.0

    # Reshape to blocks
    tensor_blocked = tensor.reshape(M // block_m, block_m, K // block_k, block_k)

    # Compute max over block dimensions
    block_max = tensor_blocked.abs().amax(dim=3).amax(dim=1)

    # Compute inverse scales
    scales = block_max / E4M3_MAX
    return torch.clamp(scales, min=1e-12)


def apply_blockwise_scaling(tensor, scales, block_m, block_k, clamp=True):
    """Apply blockwise scaling and quantize to FP8."""
    M, K = tensor.shape
    E4M3_MAX = 448.0

    # Reshape tensor and scales
    tensor_blocked = tensor.reshape(M // block_m, block_m, K // block_k, block_k)
    scales_expanded = scales.unsqueeze(1).unsqueeze(3)

    # Apply scaling
    tensor_scaled = tensor_blocked / scales_expanded

    if clamp:
        tensor_scaled = torch.clamp(tensor_scaled, min=-E4M3_MAX, max=E4M3_MAX)

    return tensor_scaled.reshape(M, K)


def dequantize_blockwise(tensor_fp8, scales, block_m, block_k):
    """Dequantize FP8 tensor with blockwise scales."""
    M, K = tensor_fp8.shape

    # Convert to float32
    tensor_f32 = tensor_fp8.to(torch.float32)

    # Reshape tensor and scales
    tensor_blocked = tensor_f32.reshape(M // block_m, block_m, K // block_k, block_k)
    scales_expanded = scales.unsqueeze(1).unsqueeze(3)

    # Apply dequantization
    tensor_dequant = tensor_blocked * scales_expanded

    return tensor_dequant.reshape(M, K)


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 attention output projection with blockwise scaling.

    Performs: output = attn_output @ o_proj_weight.T

    Scaling approach:
    - attn_output: BlockWise1x128 (per-row with 128-sized blocks in K)
    - o_proj_weight: BlockWise128x128 (128x128 blocks)
    """
    M, K = attn_output.shape
    N, K2 = o_proj_weight.shape
    assert K == K2, f"Dimension mismatch: {K} vs {K2}"

    # Convert to FP32 for scale computation
    attn_output_fp32 = attn_output.to(torch.float32)

    # Compute activation scales: BlockWise1x128 -> shape (M, K//128)
    scale_a = compute_blockwise_scales(attn_output_fp32, block_m=1, block_k=128)

    # Weight scales: BlockWise128x128
    # o_proj_weight is [N, K], we need to compute scales on the transposed version [K, N]
    weight_transposed_fp32 = o_proj_weight.T.to(torch.float32)  # [K, N]
    scale_b_transposed = compute_blockwise_scales(weight_transposed_fp32, block_m=128, block_k=128)

    # Apply scaling and quantize to FP8
    attn_output_scaled = apply_blockwise_scaling(attn_output_fp32, scale_a, block_m=1, block_k=128, clamp=True)
    attn_output_fp8 = attn_output_scaled.to(torch.float8_e4m3fn)

    # Weight scaling and quantization on transposed weight
    weight_transposed_scaled = apply_blockwise_scaling(weight_transposed_fp32, scale_b_transposed, block_m=128, block_k=128, clamp=True)
    weight_transposed_fp8 = weight_transposed_scaled.to(torch.float8_e4m3fn)

    # Dequantize both tensors
    attn_dequant = dequantize_blockwise(attn_output_fp8, scale_a, block_m=1, block_k=128)
    weight_transposed_dequant = dequantize_blockwise(weight_transposed_fp8, scale_b_transposed, block_m=128, block_k=128)

    # Perform matmul in float32: [M, K] @ [K, N] = [M, N]
    output = torch.matmul(attn_dequant, weight_transposed_dequant)

    # Convert to bfloat16
    return output.to(torch.bfloat16)
