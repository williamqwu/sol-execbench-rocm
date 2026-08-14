import torch
import math


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    is_causal: bool,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = 16
    num_key_value_heads = 4
    head_dim = 64
    num_key_value_groups = num_heads // num_key_value_heads
    scaling = 1.0

    query_states = torch.matmul(hidden_states, q_proj_weight.t())
    key_states = torch.matmul(hidden_states, k_proj_weight.t())
    value_states = torch.matmul(hidden_states, v_proj_weight.t())

    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * scaling
    sin = emb.sin() * scaling

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_half1 = query_states[..., : head_dim // 2]
    q_half2 = query_states[..., head_dim // 2 :]
    q_rotated = torch.cat((-q_half2, q_half1), dim=-1)
    query_states = (query_states * cos) + (q_rotated * sin)

    k_half1 = key_states[..., : head_dim // 2]
    k_half2 = key_states[..., head_dim // 2 :]
    k_rotated = torch.cat((-k_half2, k_half1), dim=-1)
    key_states = (key_states * cos) + (k_rotated * sin)

    if num_key_value_groups > 1:
        key_states = key_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
        ).reshape(batch_size, num_heads, seq_len, head_dim)
        value_states = value_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
        ).reshape(batch_size, num_heads, seq_len, head_dim)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    attn_output = torch.matmul(attn_output, o_proj_weight.t())

    return attn_output, attn_weights
