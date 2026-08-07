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

    # Project Q, K, V
    query_states = torch.nn.functional.linear(hidden_states, q_proj_weight)
    key_states = torch.nn.functional.linear(hidden_states, k_proj_weight)
    value_states = torch.nn.functional.linear(hidden_states, v_proj_weight)

    # Apply Q normalization (RMS norm)
    q_fp32 = query_states.to(torch.float32)
    q_variance = q_fp32.pow(2).mean(-1, keepdim=True)
    q_normed = q_fp32 * torch.rsqrt(q_variance + rms_norm_eps)
    query_states = (q_norm_weight * q_normed).to(hidden_states.dtype)

    # Apply K normalization (RMS norm)
    k_fp32 = key_states.to(torch.float32)
    k_variance = k_fp32.pow(2).mean(-1, keepdim=True)
    k_normed = k_fp32 * torch.rsqrt(k_variance + rms_norm_eps)
    key_states = (k_norm_weight * k_normed).to(hidden_states.dtype)

    # Reshape to [batch, num_heads, seq_len, head_dim]
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    # Apply YARN RoPE
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * attention_factor).unsqueeze(1).to(query_states.dtype)
    sin = (emb.sin() * attention_factor).unsqueeze(1).to(query_states.dtype)

    # Rotate half helper inline
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (rotate_half(key_states) * sin)

    # GQA via broadcasting (avoids 5x K/V materialization copy):
    # Q -> [B, nk, ng, S, D], K/V -> [B, nk, 1, S, D]; matmul broadcasts ng.
    query_states = query_states.view(batch_size, num_key_value_heads, num_key_value_groups, seq_len, head_dim)
    key_states = key_states.view(batch_size, num_key_value_heads, 1, seq_len, head_dim)
    value_states = value_states.view(batch_size, num_key_value_heads, 1, seq_len, head_dim)

    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * scaling
    # attention_mask is [B,1,S,S] -> [B,1,1,S,S] to broadcast over [B,nk,ng,S,S]
    attn_weights = attn_weights + attention_mask.unsqueeze(2)

    # Softmax
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Compute output
    attn_output = torch.matmul(attn_weights, value_states)  # [B, nk, ng, S, D]
    attn_output = attn_output.reshape(batch_size, num_attention_heads, seq_len, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)

    # Output projection
    output = torch.nn.functional.linear(attn_output, o_proj_weight)

    return output
