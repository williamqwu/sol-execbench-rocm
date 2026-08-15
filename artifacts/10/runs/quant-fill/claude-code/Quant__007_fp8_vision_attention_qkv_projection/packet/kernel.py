import torch


def _compute_and_apply_fp8_qkv(
    hidden_states_fp32,
    qkv_weight,
    qkv_bias,
):
    """Core computation matching reference exactly."""
    M, K = hidden_states_fp32.shape
    block_size_k = 128
    num_blocks_k = K // block_size_k

    # Compute activation scales (BlockWise1x128)
    hidden_blocked = hidden_states_fp32.reshape(M, num_blocks_k, block_size_k)
    block_max = hidden_blocked.abs().amax(dim=2)
    scale_x = torch.clamp(block_max / 448.0, min=1e-12)

    # Apply scaling to activations
    scale_x_expanded = scale_x.unsqueeze(2)
    hidden_scaled = hidden_blocked / scale_x_expanded
    hidden_scaled = torch.clamp(hidden_scaled, min=-448.0, max=448.0)
    qx = hidden_scaled.reshape(M, K).to(torch.float8_e4m3fn)

    # BlockWise128x128 scaling for weights
    w_fp32 = qkv_weight.T.to(torch.float32)
    M_w, K_w = w_fp32.shape
    block_size_m_w = 128
    block_size_k_w = 128
    num_blocks_m_w = M_w // block_size_m_w
    num_blocks_k_w = K_w // block_size_k_w

    # Compute weight scales
    w_blocked = w_fp32.reshape(num_blocks_m_w, block_size_m_w, num_blocks_k_w, block_size_k_w)
    w_block_max = w_blocked.abs().amax(dim=3).amax(dim=1)
    weight_scales = torch.clamp(w_block_max / 448.0, min=1e-12)

    # Apply scaling to weights
    weight_scales_expanded = weight_scales.unsqueeze(1).unsqueeze(3)
    w_scaled = w_blocked / weight_scales_expanded
    w_scaled = torch.clamp(w_scaled, min=-448.0, max=448.0)
    qw = w_scaled.reshape(M_w, K_w).T.to(torch.float8_e4m3fn)

    # Dequantize activations
    qx_fp32 = qx.to(torch.float32)
    qx_blocked = qx_fp32.reshape(M, num_blocks_k, block_size_k)
    a_f32 = (qx_blocked * scale_x_expanded).reshape(M, K)

    # Dequantize weights
    qw_t_fp32 = qw.T.to(torch.float32)
    qw_t_blocked = qw_t_fp32.reshape(num_blocks_m_w, block_size_m_w, num_blocks_k_w, block_size_k_w)
    b_f32 = (qw_t_blocked * weight_scales_expanded).reshape(M_w, K_w)

    # Matrix multiplication in float32
    qkv = a_f32 @ b_f32

    # Add bias
    if qkv_bias is not None and qkv_bias.numel():
        qkv = qkv + qkv_bias

    return qkv


# Compile the core computation
_compiled_fn = torch.compile(_compute_and_apply_fp8_qkv, mode="reduce-overhead")


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
):
    """
    Optimized FP8-quantized QKV projection.

    Args:
        hidden_states: Input tensor (seq_len, hidden_size) in bfloat16
        qkv_weight: QKV projection weight (qkv_out_features, hidden_size) in bfloat16
        qkv_bias: QKV projection bias (qkv_out_features,) in bfloat16

    Returns:
        Tuple of (query_states, key_states, value_states)
    """
    # Constants
    num_heads = 16
    head_dim = 96
    seq_length = hidden_states.shape[0]

    # Convert to FP32 for scaling operations
    hidden_states_fp32 = hidden_states.to(torch.float32)

    # Use compiled version of the core computation
    qkv = _compiled_fn(hidden_states_fp32, qkv_weight, qkv_bias)

    # Convert to bfloat16
    qkv = qkv.to(torch.bfloat16)

    # Reshape and split into Q, K, V
    qkv = qkv.view(seq_length, 3, num_heads, head_dim)
    query_states, key_states, value_states = qkv.unbind(dim=1)

    # Ensure contiguous output
    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()

    return query_states, key_states, value_states
