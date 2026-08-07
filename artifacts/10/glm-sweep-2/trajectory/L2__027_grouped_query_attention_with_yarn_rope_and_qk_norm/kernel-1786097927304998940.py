import torch

_RMSNORM = None
_ROPE = None

def _get_compiled():
    global _RMSNORM, _ROPE
    if _RMSNORM is None:
        @torch.compile(fullgraph=True, dynamic=True)
        def rmsnorm(x, weight, eps):
            xf = x.float()
            var = xf.pow(2).mean(-1, keepdim=True)
            return (weight * (xf * torch.rsqrt(var + eps))).to(x.dtype)
        @torch.compile(fullgraph=True, dynamic=True)
        def apply_rope(q, k, cos, sin, half):
            q1 = q[..., :half]; q2 = q[..., half:]
            k1 = k[..., :half]; k2 = k[..., half:]
            qo = (q * cos) + (torch.cat((-q2, q1), dim=-1) * sin)
            ko = (k * cos) + (torch.cat((-k2, k1), dim=-1) * sin)
            return qo, ko
        _RMSNORM = rmsnorm
        _ROPE = apply_rope
    return _RMSNORM, _ROPE


def run(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    rms_norm_eps: float,
    attention_factor: float,
    scaling: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 40
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = 5
    half_head_dim = head_dim // 2
    dtype = hidden_states.dtype

    rmsnorm, apply_rope = _get_compiled()

    # Project Q, K, V (eager — preserve reduction order)
    query_states = torch.nn.functional.linear(hidden_states, q_proj_weight)
    key_states = torch.nn.functional.linear(hidden_states, k_proj_weight)
    value_states = torch.nn.functional.linear(hidden_states, v_proj_weight)

    # RMS norm (compiled/fused)
    query_states = rmsnorm(query_states, q_norm_weight, rms_norm_eps)
    key_states = rmsnorm(key_states, k_norm_weight, rms_norm_eps)

    # Reshape
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    # YARN RoPE cos/sin
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * attention_factor).unsqueeze(1).to(dtype)
    sin = (emb.sin() * attention_factor).unsqueeze(1).to(dtype)

    # RoPE (compiled/fused)
    query_states, key_states = apply_rope(query_states, key_states, cos, sin, half_head_dim)

    # Repeat KV for GQA
    key_states = key_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    value_states = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)

    # Attention (eager — preserve reduction order)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)

    output = torch.nn.functional.linear(attn_output, o_proj_weight)
    return output
