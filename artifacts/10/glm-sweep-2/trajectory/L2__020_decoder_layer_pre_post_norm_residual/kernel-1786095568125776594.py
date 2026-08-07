import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    num_attention_heads = 64
    num_key_value_heads = 8
    head_dim = 96
    num_key_value_groups = num_attention_heads // num_key_value_heads
    half_dim = head_dim // 2

    batch_size, seq_len, _ = hidden_states.shape

    def rms_norm(x, weight, eps):
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + eps)
        return (weight * x_normed.to(input_dtype))

    residual1 = hidden_states
    hidden_states = rms_norm(hidden_states, input_layernorm_weight, rms_norm_eps)

    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)

    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)

    q1 = query_states[..., :half_dim]
    q2 = query_states[..., half_dim:]
    query_states = torch.cat([q1 * cos_expanded - q2 * sin_expanded, q1 * sin_expanded + q2 * cos_expanded], dim=-1)

    k1 = key_states[..., :half_dim]
    k2 = key_states[..., half_dim:]
    key_states = torch.cat([k1 * cos_expanded - k2 * sin_expanded, k1 * sin_expanded + k2 * cos_expanded], dim=-1)

    key_states = key_states[:, :, None, :, :].expand(batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim).reshape(batch_size, num_key_value_heads * num_key_value_groups, seq_len, head_dim)
    value_states = value_states[:, :, None, :, :].expand(batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim).reshape(batch_size, num_key_value_heads * num_key_value_groups, seq_len, head_dim)

    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, attn_mask=attention_mask, is_causal=False
    )

    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    attn_output = F.linear(attn_output, o_proj_weight)

    hidden_states = residual1 + attn_output

    residual2 = hidden_states
    hidden_states = rms_norm(hidden_states, post_attention_layernorm_weight, rms_norm_eps)

    gate = F.silu(F.linear(hidden_states, gate_proj_weight))
    up = F.linear(hidden_states, up_proj_weight)
    hidden_states = F.linear(gate * up, down_proj_weight)

    output = residual2 + hidden_states
    return output


run = torch.compile(run, mode="reduce-overhead")
