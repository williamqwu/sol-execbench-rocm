import torch


def compute_scales_1x128(x_fp32):
    """Compute 1x128 blockwise scales."""
    M, K = x_fp32.shape
    assert K % 128 == 0

    # Reshape and compute max per block
    x_blocked = x_fp32.view(M, K // 128, 128)
    block_max = x_blocked.abs().amax(dim=2)
    scales = (block_max / 448.0).clamp(min=1e-12)

    return scales


def compute_scales_128x128(x_fp32):
    """Compute 128x128 blockwise scales."""
    M, K = x_fp32.shape

    # Pad if necessary
    pad_m = (128 - M % 128) % 128
    pad_k = (128 - K % 128) % 128

    if pad_m > 0 or pad_k > 0:
        x_fp32 = torch.nn.functional.pad(x_fp32, (0, pad_k, 0, pad_m))
        M_padded, K_padded = x_fp32.shape
    else:
        M_padded, K_padded = M, K

    # Reshape to blocks
    x_blocked = x_fp32.view(M_padded // 128, 128, K_padded // 128, 128)
    block_max = x_blocked.abs().amax(dim=3).amax(dim=1)
    scales = (block_max / 448.0).clamp(min=1e-12)

    return scales


def apply_scaling_1x128(tensor, scales, inverse=False, clamp=False):
    """Apply 1x128 blockwise scaling."""
    M, K = tensor.shape
    new_shape = (M, K // 128, 128)
    tensor_blocked = tensor.view(new_shape)
    scales_expanded = scales.unsqueeze(2)

    if inverse:
        result = tensor_blocked * scales_expanded
    else:
        result = tensor_blocked / scales_expanded
        if clamp:
            result = torch.clamp(result, min=-448.0, max=448.0)

    return result.view(M, K)


def apply_scaling_128x128(tensor, scales, inverse=False, clamp=False):
    """Apply 128x128 blockwise scaling."""
    M_orig, K_orig = tensor.shape

    # Pad if necessary
    pad_m = (128 - M_orig % 128) % 128
    pad_k = (128 - K_orig % 128) % 128

    if pad_m > 0 or pad_k > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_k, 0, pad_m))

    M, K = tensor.shape
    new_shape = (M // 128, 128, K // 128, 128)
    tensor_blocked = tensor.view(new_shape)
    scales_expanded = scales.unsqueeze(1).unsqueeze(3)

    if inverse:
        result = tensor_blocked * scales_expanded
    else:
        result = tensor_blocked / scales_expanded
        if clamp:
            result = torch.clamp(result, min=-448.0, max=448.0)

    result = result.view(M, K)

    # Remove padding
    if pad_m > 0 or pad_k > 0:
        result = result[:M_orig, :K_orig]

    return result


def fp8_linear(x, weight, bias, output_dtype=torch.bfloat16):
    """FP8 linear layer with blockwise quantization."""
    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)

    # Compute scales for activations (1x128)
    scale_x = compute_scales_1x128(x_fp32)

    # Compute scales for weights (128x128) on transposed weight
    weight_t = weight_fp32.T.contiguous()  # (K, N)
    scales_w = compute_scales_128x128(weight_t)

    # Quantize
    x_scaled = apply_scaling_1x128(x_fp32, scale_x, inverse=False, clamp=True)
    weight_t_scaled = apply_scaling_128x128(weight_t, scales_w, inverse=False, clamp=True)

    qx = x_scaled.to(torch.float8_e4m3fn)
    qw_t = weight_t_scaled.to(torch.float8_e4m3fn)

    # Dequantize
    x_dequant = apply_scaling_1x128(qx.to(torch.float32), scale_x, inverse=True)
    w_t_dequant = apply_scaling_128x128(qw_t.to(torch.float32), scales_w, inverse=True)

    # Matmul: (M, K) @ (K, N) = (M, N)
    output = x_dequant @ w_t_dequant

    # Add bias
    if bias is not None and bias.numel() > 0:
        output = output + bias

    return output.to(output_dtype)


def rms_norm(hidden_states, weight, eps):
    """RMSNorm on last dimension."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return (weight * hidden_states).to(input_dtype)


def _run_impl(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    v_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rms_norm_eps: float,
):
    """FP8 QKV projection with Q/K RMSNorm normalization."""
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 60
    num_key_value_heads = 8
    head_dim = 128

    # Flatten batch and sequence dimensions
    hidden_states_flat = hidden_states.reshape(-1, hidden_size)

    # QKV projections with FP8
    query_states = fp8_linear(hidden_states_flat, q_weight, q_bias)
    key_states = fp8_linear(hidden_states_flat, k_weight, k_bias)
    value_states = fp8_linear(hidden_states_flat, v_weight, v_bias)

    # Reshape to heads
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)

    # Apply RMSNorm to Q and K
    query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps)
    key_states = rms_norm(key_states, k_norm_weight, rms_norm_eps)

    # Transpose to (batch, heads, seq, head_dim)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    return query_states, key_states, value_states


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    v_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rms_norm_eps: float,
):
    """Entry point."""
    return _run_impl(
        hidden_states, q_weight, k_weight, v_weight,
        q_bias, k_bias, v_bias,
        q_norm_weight, k_norm_weight, rms_norm_eps
    )
