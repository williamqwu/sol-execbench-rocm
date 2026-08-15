import torch


def apply_scaling_blockwise(tensor, scales, block_size_m, block_size_k, inverse=True):
    """
    Apply blockwise scaling to a tensor, matching the reference BlockwiseScaler.

    Args:
        tensor: (M, K) tensor
        scales: Scale factors with shape depending on block sizes
        block_size_m: Block size in M dimension (or None for full dimension)
        block_size_k: Block size in K dimension (or None for full dimension)
        inverse: If True, multiply by scales (dequantization)

    Returns:
        Scaled tensor (M, K)
    """
    M, K = tensor.shape

    # Reshape (M, K) -> (M//block_size_m, block_size_m, K//block_size_k, block_size_k)
    new_shape = (
        M // block_size_m,
        block_size_m,
        K // block_size_k,
        block_size_k,
    )
    tensor_blocked = tensor.reshape(new_shape)

    # Expand scales: (M//block_size_m, K//block_size_k) -> (M//block_size_m, 1, K//block_size_k, 1)
    scales_expanded = scales.unsqueeze(1).unsqueeze(3)

    # Apply scaling
    if inverse:
        tensor_scaled = tensor_blocked * scales_expanded
    else:
        tensor_scaled = tensor_blocked / scales_expanded

    return tensor_scaled.reshape(M, K)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
):
    """
    FP8 MLA attention output projection.

    Args:
        hidden_states: Input tensor [batch_size, seq_len, 16384] in FP8
        weight: Weight matrix [7168, 16384] in FP8
        scale_x: Activation scales [batch_size, seq_len, 128] for BlockWise1x128
        scale_w: Weight scales [128, 56] for BlockWise128x128

    Returns:
        Output tensor [batch_size, seq_len, 7168] in bfloat16
    """
    batch_size, seq_len, input_dim = hidden_states.shape
    hidden_size = weight.shape[0]

    # Reshape to 2D for GEMM: (B*L, input_dim)
    x = hidden_states.view(-1, input_dim)  # (B*L, 16384)
    M = x.shape[0]  # B*L
    K = input_dim
    N = hidden_size

    # Reshape scales for 2D GEMM
    scale_x_2d = scale_x.view(M, -1)  # (B*L, 128)

    # Dequantize activations: BlockWise1x128
    # x is (M, K), scale_x_2d is (M, K//128)
    x_f32 = apply_scaling_blockwise(
        x.to(torch.float32), scale_x_2d,
        block_size_m=1, block_size_k=128, inverse=True
    )

    # Dequantize weights: BlockWise128x128
    # weight is (N, K) = (7168, 16384), scale_w is (K//128, N//128) = (128, 56)
    # The reference transposes scale_w to (N//128, K//128) = (56, 128) before applying
    scale_w_t = scale_w.T.contiguous()  # (56, 128)
    w_f32 = apply_scaling_blockwise(
        weight.to(torch.float32), scale_w_t,
        block_size_m=128, block_size_k=128, inverse=True
    )  # (N, K) = (7168, 16384)

    # Matmul in float32: (M, K) @ (K, N) = (M, N)
    output = x_f32 @ w_f32.T

    # Convert to bfloat16
    output = output.to(torch.bfloat16)

    # Reshape back to 3D: (B, L, hidden_size)
    output = output.view(batch_size, seq_len, hidden_size)

    return output
