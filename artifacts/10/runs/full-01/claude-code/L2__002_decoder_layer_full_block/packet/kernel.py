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
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim ** -0.5

    batch_size, seq_len, _ = hidden_states.shape
    residual = hidden_states

    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + rms_norm_eps)
    hidden_states = input_layernorm_weight * x.to(hidden_states.dtype)

    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)

    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)

    q1 = query_states[..., : head_dim // 2]
    q2 = query_states[..., head_dim // 2:]
    q_rotated = torch.cat((-q2, q1), dim=-1)
    query_states = (query_states * cos_expanded) + (q_rotated * sin_expanded)

    k1 = key_states[..., : head_dim // 2]
    k2 = key_states[..., head_dim // 2:]
    k_rotated = torch.cat((-k2, k1), dim=-1)
    key_states = (key_states * cos_expanded) + (k_rotated * sin_expanded)

    key_states = key_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)

    value_states = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().view(
        batch_size, seq_len, num_attention_heads * head_dim
    )

    attn_output = F.linear(attn_output, o_proj_weight)
    hidden_states = residual + attn_output

    residual = hidden_states

    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + rms_norm_eps)
    hidden_states = post_attention_layernorm_weight * x.to(hidden_states.dtype)

    gate = F.silu(F.linear(hidden_states, gate_proj_weight))
    up = F.linear(hidden_states, up_proj_weight)
    mlp_output = F.linear(gate * up, down_proj_weight)

    output = residual + mlp_output

    return output
