import torch
import torch.nn.functional as F

_SCALING = 128 ** -0.5
_mask_cache = {}
_scale_cache = {}


def _get_mask(seq_len, device, dtype):
    key = (seq_len, device.index, dtype)
    m = _mask_cache.get(key)
    if m is None or m.device != device:
        m = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype),
            diagonal=1,
        )
        _mask_cache[key] = m
    return m


def _get_scale(device, dtype):
    key = (device.index, dtype)
    s = _scale_cache.get(key)
    if s is None or s.device != device:
        s = torch.tensor(_SCALING, device=device, dtype=dtype)
        _scale_cache[key] = s
    return s


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
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = 4
    scaling = _SCALING

    batch_size, seq_len, _ = hidden_states.shape

    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)

    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)

    q1 = query_states[..., : head_dim // 2]
    q2 = query_states[..., head_dim // 2 :]
    query_rotated = torch.cat((-q2, q1), dim=-1)
    query_states = (query_states * cos_expanded) + (query_rotated * sin_expanded)

    k1 = key_states[..., : head_dim // 2]
    k2 = key_states[..., head_dim // 2 :]
    key_rotated = torch.cat((-k2, k1), dim=-1)
    key_states = (key_states * cos_expanded) + (key_rotated * sin_expanded)

    key_states = key_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    value_states = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)

    attn_scores = torch.matmul(query_states, key_states.transpose(2, 3))
    causal_mask = _get_mask(seq_len, hidden_states.device, hidden_states.dtype)
    attn_weights = torch.addcmul(causal_mask, attn_scores, _get_scale(hidden_states.device, hidden_states.dtype))

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)

    output = F.linear(attn_output, o_proj_weight)
    return output
