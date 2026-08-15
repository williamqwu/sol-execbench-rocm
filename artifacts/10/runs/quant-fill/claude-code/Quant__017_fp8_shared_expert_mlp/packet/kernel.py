import torch
import torch.nn.functional as F


E4M3_MAX = 448.0


def _compute_blockwise_1x128_scales(tensor: torch.Tensor) -> torch.Tensor:
    """Compute 1x128 blockwise scales (rowwise in M, 128-blocks in K)."""
    M, K = tensor.shape

    # Reshape to [M, K//128, 128]
    tensor_blocked = tensor.reshape(M, K // 128, 128)

    # Max over the 128-sized blocks: [M, K//128]
    block_max = tensor_blocked.abs().amax(dim=2)
    scales = (block_max / E4M3_MAX).clamp(min=1e-12)

    return scales


def _apply_blockwise_1x128_scaling(
    tensor: torch.Tensor,
    scales: torch.Tensor,
    inverse: bool = False,
    clamp_to_fp8_range: bool = False,
) -> torch.Tensor:
    """Apply 1x128 blockwise scaling."""
    M, K = tensor.shape

    # Reshape to [M, K//128, 128]
    tensor_blocked = tensor.reshape(M, K // 128, 128)
    scales_expanded = scales.unsqueeze(2)  # [M, K//128, 1]

    if inverse:
        tensor_scaled = tensor_blocked * scales_expanded
    else:
        tensor_scaled = tensor_blocked / scales_expanded
        if clamp_to_fp8_range:
            tensor_scaled = tensor_scaled.clamp(-E4M3_MAX, E4M3_MAX)

    return tensor_scaled.reshape(M, K)


def _apply_blockwise_128x128_scaling(
    tensor: torch.Tensor,
    scales: torch.Tensor,
    inverse: bool = False,
    clamp_to_fp8_range: bool = False,
) -> torch.Tensor:
    """Apply 128x128 blockwise scaling."""
    M, K = tensor.shape

    # Reshape to [M//128, 128, K//128, 128]
    tensor_blocked = tensor.reshape(M // 128, 128, K // 128, 128)
    scales_expanded = scales.unsqueeze(1).unsqueeze(3)  # [M//128, 1, K//128, 1]

    if inverse:
        tensor_scaled = tensor_blocked * scales_expanded
    else:
        tensor_scaled = tensor_blocked / scales_expanded
        if clamp_to_fp8_range:
            tensor_scaled = tensor_scaled.clamp(-E4M3_MAX, E4M3_MAX)

    return tensor_scaled.reshape(M, K)


def _fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 linear layer with blockwise scaling.

    Args:
        x: Input tensor [M, K] in BF16
        weight: Weight tensor [N, K] in BF16
        weight_scales: Pre-computed weight scales [K//128, N//128]

    Returns:
        Output tensor [M, N] in BF16
    """
    M, K = x.shape
    N, _ = weight.shape

    # Step 1: Compute activation scales dynamically (1x128 blockwise)
    x_fp32 = x.to(torch.float32)
    scale_x = _compute_blockwise_1x128_scales(x_fp32)  # [M, K//128]

    # Step 2: Apply scaling and quantize input
    x_scaled = _apply_blockwise_1x128_scaling(
        x_fp32, scale_x, inverse=False, clamp_to_fp8_range=True
    )
    qx = x_scaled.to(torch.float8_e4m3fn)  # [M, K]

    # Step 3: Apply scaling and quantize weight
    weight_fp32 = weight.T.to(torch.float32)  # [K, N]
    weight_scaled = _apply_blockwise_128x128_scaling(
        weight_fp32, weight_scales, inverse=False, clamp_to_fp8_range=True
    )
    qw = weight_scaled.T.to(torch.float8_e4m3fn)  # [N, K]

    # Step 4: Transpose weight scales for GEMM
    # Scales from compute_scales are [K//128, N//128], need [N//128, K//128]
    scale_w_cublas = weight_scales.T.contiguous()

    # Step 5: FP8 GEMM - dequantize then matmul
    # Dequantize X: [M, K]
    qx_f32 = qx.to(torch.float32)
    a_f32 = _apply_blockwise_1x128_scaling(qx_f32, scale_x, inverse=True)

    # Dequantize W: [N, K]
    qw_f32 = qw.to(torch.float32)
    # For W, need to dequantize [N, K] with scales [N//128, K//128]
    w_f32 = _apply_blockwise_128x128_scaling(qw_f32, scale_w_cublas, inverse=True)

    # Matmul in float32
    output = a_f32 @ w_f32.T

    return output.to(torch.bfloat16)


@torch.no_grad()
def run(
    x: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    gate_proj_weight_scales: torch.Tensor,
    up_proj_weight: torch.Tensor,
    up_proj_weight_scales: torch.Tensor,
    down_proj_weight: torch.Tensor,
    down_proj_weight_scales: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 Shared Expert MLP forward pass.

    Computation:
        gate = silu(gate_proj(x))  # SiLU NOT quantized
        up = up_proj(x)
        output = down_proj(gate * up)

    Args:
        x: Input tensor [num_tokens, hidden_size] in BF16
        gate_proj_weight: Gate projection weights [intermediate_size, hidden_size]
        gate_proj_weight_scales: FP8 scales for gate weights [hidden_size//128, intermediate_size//128]
        up_proj_weight: Up projection weights [intermediate_size, hidden_size]
        up_proj_weight_scales: FP8 scales for up weights [hidden_size//128, intermediate_size//128]
        down_proj_weight: Down projection weights [hidden_size, intermediate_size]
        down_proj_weight_scales: FP8 scales for down weights [intermediate_size//128, hidden_size//128]

    Returns:
        Output tensor [num_tokens, hidden_size] in BF16
    """
    # FP8 gate projection: [num_tokens, hidden_size] @ [hidden_size, intermediate_size] -> [num_tokens, intermediate_size]
    gate_output = _fp8_linear(x, gate_proj_weight, gate_proj_weight_scales)

    # SiLU activation (NOT quantized, remains in BF16)
    gate_activated = F.silu(gate_output)

    # FP8 up projection: [num_tokens, hidden_size] @ [hidden_size, intermediate_size] -> [num_tokens, intermediate_size]
    up_output = _fp8_linear(x, up_proj_weight, up_proj_weight_scales)

    # Element-wise multiplication (NOT quantized, remains in BF16)
    intermediate = gate_activated * up_output

    # FP8 down projection: [num_tokens, intermediate_size] @ [intermediate_size, hidden_size] -> [num_tokens, hidden_size]
    output = _fp8_linear(intermediate, down_proj_weight, down_proj_weight_scales)

    return output
