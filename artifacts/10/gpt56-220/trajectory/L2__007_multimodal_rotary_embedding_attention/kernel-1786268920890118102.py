import torch
import torch.nn.functional as F

@torch.compile
def _apply_rope(query_states, key_states, cos, sin):
    c = torch.cat((cos[0, ..., :32], cos[1, ..., 32:80], cos[2, ..., 80:]), dim=-1).unsqueeze(1)
    s = torch.cat((sin[0, ..., :32], sin[1, ..., 32:80], sin[2, ..., 80:]), dim=-1).unsqueeze(1)
    qrot = torch.cat((-query_states[..., 64:], query_states[..., :64]), dim=-1)
    krot = torch.cat((-key_states[..., 64:], key_states[..., :64]), dim=-1)
    # Returning the products materializes them as bf16, matching eager's
    # rounding boundary before the final additions.
    return query_states * c, qrot * s, key_states * c, krot * s

@torch.compile
def _finish_rope(query_cos, query_sin, key_cos, key_sin):
    return query_cos + query_sin, key_cos + key_sin

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
    o_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """
    Grouped Query Attention with Multi-modal 3D Rotary Position Embeddings.
    
    Args:
        hidden_states: [batch_size, seq_len, hidden_size]
        q_weight: [num_heads * head_dim, hidden_size]
        q_bias: [num_heads * head_dim]
        k_weight: [num_kv_heads * head_dim, hidden_size]
        k_bias: [num_kv_heads * head_dim]
        v_weight: [num_kv_heads * head_dim, hidden_size]
        v_bias: [num_kv_heads * head_dim]
        o_weight: [hidden_size, num_heads * head_dim]
        cos: [3, batch_size, seq_len, head_dim]
        sin: [3, batch_size, seq_len, head_dim]
        attention_mask: [batch_size, 1, seq_len, seq_len]
    
    Returns:
        output: [batch_size, seq_len, hidden_size]
    """
    # Constants
    num_heads = 28
    num_kv_heads = 4
    num_kv_groups = 7
    head_dim = 128
    scaling = head_dim ** -0.5
    mrope_section = [16, 24, 24]  # Channel splits for temporal/height/width
    
    bsz, q_len, _ = hidden_states.size()
    
    # Project to Q, K, V using linear operations
    query_states = F.linear(hidden_states, q_weight, q_bias)
    key_states = F.linear(hidden_states, k_weight, k_bias)
    value_states = F.linear(hidden_states, v_weight, v_bias)
    
    # Reshape to [batch, num_heads, seq_len, head_dim]
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # Compilation removes several memory-bound concatenation kernels, but its
    # fixed launch overhead is not worthwhile for small token batches.
    if bsz * q_len <= 256:
        rope_cos = torch.cat((cos[0, ..., :32], cos[1, ..., 32:80], cos[2, ..., 80:]), dim=-1).unsqueeze(1)
        rope_sin = torch.cat((sin[0, ..., :32], sin[1, ..., 32:80], sin[2, ..., 80:]), dim=-1).unsqueeze(1)
        query_rot = torch.cat((-query_states[..., 64:], query_states[..., :64]), dim=-1)
        key_rot = torch.cat((-key_states[..., 64:], key_states[..., :64]), dim=-1)
    else:
        query_cos, query_sin, key_cos, key_sin = _apply_rope(query_states, key_states, cos, sin)
        if bsz * q_len >= 2048:
            query_states, key_states = _finish_rope(query_cos, query_sin, key_cos, key_sin)
        else:
            query_states = query_cos + query_sin
            key_states = key_cos + key_sin
    if bsz * q_len <= 256:
        query_states = query_states * rope_cos + query_rot * rope_sin
        key_states = key_states * rope_cos + key_rot * rope_sin

    key_states = key_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, num_kv_groups, q_len, head_dim
    ).reshape(bsz, num_heads, q_len, head_dim)
    value_states = value_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, num_kv_groups, q_len, head_dim
    ).reshape(bsz, num_heads, q_len, head_dim)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
    output = F.linear(attn_output, o_weight)
    
    return output
