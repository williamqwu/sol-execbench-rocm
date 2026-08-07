import torch

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

    # Fuse QKV projection: one matmul instead of three.
    qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    qkv = torch.nn.functional.linear(hidden_states, qkv_weight)
    q_out, k_out, v_out = qkv.split([hidden_size, 1024, 1024], dim=-1)

    # Apply Q normalization (RMS norm) — fp32 accumulation
    q_fp32 = q_out.float()
    q_variance = q_fp32.pow(2).mean(-1, keepdim=True)
    q_out = (q_norm_weight * (q_fp32 * torch.rsqrt(q_variance + rms_norm_eps))).to(dtype)

    # Apply K normalization (RMS norm) — fp32 accumulation
    k_fp32 = k_out.float()
    k_variance = k_fp32.pow(2).mean(-1, keepdim=True)
    k_out = (k_norm_weight * (k_fp32 * torch.rsqrt(k_variance + rms_norm_eps))).to(dtype)

    # Reshape to [batch, num_heads, seq_len, head_dim]
    query_states = q_out.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = k_out.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = v_out.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    # Apply YARN RoPE
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * attention_factor).unsqueeze(1).to(dtype)
    sin = (emb.sin() * attention_factor).unsqueeze(1).to(dtype)

    q1 = query_states[..., :half_head_dim]
    q2 = query_states[..., half_head_dim:]
    query_states = (query_states * cos) + (torch.cat((-q2, q1), dim=-1) * sin)
    k1 = key_states[..., :half_head_dim]
    k2 = key_states[..., half_head_dim:]
    key_states = (key_states * cos) + (torch.cat((-k2, k1), dim=-1) * sin)

    # Repeat KV for GQA
    key_states = key_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)
    value_states = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len, head_dim)

    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)

    # Compute output
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)

    # Output projection
    output = torch.nn.functional.linear(attn_output, o_proj_weight)
    return output
