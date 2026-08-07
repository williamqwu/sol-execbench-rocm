import torch
import torch.nn.functional as F


@torch.compile()
def _run_compiled(
    hidden_states,
    q_proj_weight, q_proj_bias,
    k_proj_weight, k_proj_bias,
    v_proj_weight, v_proj_bias,
    o_proj_weight,
    q_norm_weight, k_norm_weight,
    cos, sin,
    rms_norm_eps,
):
    batch_size, seq_length, _ = hidden_states.shape
    num_attention_heads = 96
    num_key_value_heads = 8
    head_dim = 128

    # Q/K/V projections
    query_states = F.linear(hidden_states, q_proj_weight, q_proj_bias)
    key_states = F.linear(hidden_states, k_proj_weight, k_proj_bias)
    value_states = F.linear(hidden_states, v_proj_weight, v_proj_bias)

    # Reshape to separate heads
    query_states = query_states.view(batch_size, seq_length, num_attention_heads, head_dim)
    key_states = key_states.view(batch_size, seq_length, num_key_value_heads, head_dim)
    value_states = value_states.view(batch_size, seq_length, num_key_value_heads, head_dim)

    # Apply QK RMSNorm
    def rms_norm(x, weight):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + rms_norm_eps)
        return (weight * x).to(input_dtype)

    query_states = rms_norm(query_states, q_norm_weight)
    key_states = rms_norm(key_states, k_norm_weight)

    # Transpose to [batch, heads, seq, head_dim]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # Apply RoPE
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)

    q1, q2 = query_states[..., :64], query_states[..., 64:]
    q_rot_half = torch.cat((-q2, q1), dim=-1)
    query_states = (query_states * cos_expanded) + (q_rot_half * sin_expanded)

    k1, k2 = key_states[..., :64], key_states[..., 64:]
    k_rot_half = torch.cat((-k2, k1), dim=-1)
    key_states = (key_states * cos_expanded) + (k_rot_half * sin_expanded)

    # Fused GQA attention
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, is_causal=True, enable_gqa=True,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_length, num_attention_heads * head_dim)

    output = F.linear(attn_output, o_proj_weight, None)
    return output


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_norm_eps: float,
):
    return _run_compiled(
        hidden_states,
        q_proj_weight, q_proj_bias,
        k_proj_weight, k_proj_bias,
        v_proj_weight, v_proj_bias,
        o_proj_weight,
        q_norm_weight, k_norm_weight,
        cos, sin,
        rms_norm_eps,
    )
