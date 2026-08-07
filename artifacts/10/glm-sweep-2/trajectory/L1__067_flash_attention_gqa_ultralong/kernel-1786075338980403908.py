import torch
import torch.nn.functional as F

_NUM_HEADS = 32
_NUM_KV = 8
_HEAD_DIM = 128
_NGROUP = 4
_SCALING = _HEAD_DIM ** -0.5


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
):
    num_attention_heads = _NUM_HEADS
    num_key_value_heads = _NUM_KV
    head_dim = _HEAD_DIM
    num_key_value_groups = _NGROUP
    scaling = _SCALING

    batch_size, seq_len, _ = hidden_states.shape

    # Fused QKV projection (one matmul instead of three); bit-exact vs separate
    qkv = F.linear(hidden_states, torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0))
    q_off = num_attention_heads * head_dim
    kv_off = num_key_value_heads * head_dim
    query_states = qkv[..., :q_off]
    key_states = qkv[..., q_off:q_off + kv_off]
    value_states = qkv[..., q_off + kv_off:]

    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    # Apply RoPE: result halves are [q1*cos - q2*sin, q2*cos + q1*sin]
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    half = head_dim // 2

    q1 = query_states[..., :half]
    q2 = query_states[..., half:]
    query_states = query_states * cos_expanded
    query_states[..., :half] = q1 * cos_expanded[..., :half] - q2 * sin_expanded[..., :half]
    query_states[..., half:] = q2 * cos_expanded[..., half:] + q1 * sin_expanded[..., half:]

    k1 = key_states[..., :half]
    k2 = key_states[..., half:]
    key_states = key_states * cos_expanded
    key_states[..., :half] = k1 * cos_expanded[..., :half] - k2 * sin_expanded[..., :half]
    key_states[..., half:] = k2 * cos_expanded[..., half:] + k1 * sin_expanded[..., half:]

    key_states = key_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    value_states = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype),
        diagonal=1
    )
    attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)

    output = F.linear(attn_output, o_proj_weight)
    return output
