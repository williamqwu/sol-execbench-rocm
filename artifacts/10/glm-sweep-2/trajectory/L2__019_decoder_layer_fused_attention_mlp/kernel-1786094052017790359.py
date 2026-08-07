import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    o_proj_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    hidden_size = 3584
    num_heads = 28
    num_kv_heads = 4
    head_dim = 128
    num_kv_groups = num_heads // num_kv_heads
    mrope_section = [16, 24, 24]
    scaling = head_dim ** -0.5

    batch_size, seq_len, _ = hidden_states.shape

    residual = hidden_states

    # 1. Pre-attention RMSNorm (fused inline)
    hidden_fp32 = hidden_states.to(torch.float32)
    variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
    hidden_normed = hidden_fp32 * torch.rsqrt(variance + rms_norm_eps)
    normed = (input_layernorm_weight * hidden_normed).to(hidden_states.dtype)

    # 2. QKV projections
    query_states = F.linear(normed, q_proj_weight, q_proj_bias)
    key_states = F.linear(normed, k_proj_weight, k_proj_bias)
    value_states = F.linear(normed, v_proj_weight, v_proj_bias)

    # Reshape for multi-head attention
    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # 3. Apply multimodal 3D RoPE
    mrope_section_doubled = [s * 2 for s in mrope_section]  # [32, 48, 48]

    cos_parts = []
    sin_parts = []
    for i, section_size in enumerate(mrope_section_doubled):
        cos_i = rope_cos[i % 3]
        sin_i = rope_sin[i % 3]
        start_idx = sum(mrope_section_doubled[:i])
        end_idx = start_idx + section_size
        cos_parts.append(cos_i[..., start_idx:end_idx])
        sin_parts.append(sin_i[..., start_idx:end_idx])

    cos_combined = torch.cat(cos_parts, dim=-1)[:, :seq_len, :].unsqueeze(1)
    sin_combined = torch.cat(sin_parts, dim=-1)[:, :seq_len, :].unsqueeze(1)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    query_states = (query_states * cos_combined) + (rotate_half(query_states) * sin_combined)
    key_states = (key_states * cos_combined) + (rotate_half(key_states) * sin_combined)

    # 4-8. Attention via SDPA (math backend for fp32). enable_gqa avoids
    # materializing the 7x-repeated KV tensors. is_causal avoids building the
    # (seq,seq) mask and lets the fused softmax handle masking internally.
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states,
        is_causal=True, enable_gqa=True,
    )
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, hidden_size)
    attn_output = F.linear(attn_output, o_proj_weight)

    # 9. First residual
    hidden_states = residual + attn_output

    # 10. Post-attention RMSNorm
    residual = hidden_states
    hidden_fp32 = hidden_states.to(torch.float32)
    variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
    hidden_normed = hidden_fp32 * torch.rsqrt(variance + rms_norm_eps)
    normed = (post_attention_layernorm_weight * hidden_normed).to(hidden_states.dtype)

    # 11. SwiGLU MLP
    gate_output = F.linear(normed, gate_proj_weight)
    up_output = F.linear(normed, up_proj_weight)
    intermediate = F.silu(gate_output) * up_output
    hidden_states = F.linear(intermediate, down_proj_weight)

    # 12. Second residual
    output = residual + hidden_states
    return output
