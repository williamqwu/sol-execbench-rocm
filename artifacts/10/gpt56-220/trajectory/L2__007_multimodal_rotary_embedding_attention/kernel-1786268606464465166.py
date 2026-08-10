import torch
import torch.nn.functional as F

@torch.compile
def _apply_rope(query_states, key_states, cos, sin):
    c = torch.cat((cos[0, ..., :32], cos[1, ..., 32:80], cos[2, ..., 80:]), dim=-1).unsqueeze(1)
    s = torch.cat((sin[0, ..., :32], sin[1, ..., 32:80], sin[2, ..., 80:]), dim=-1).unsqueeze(1)
    qrot = torch.cat((-query_states[..., 64:], query_states[..., :64]), dim=-1)
    krot = torch.cat((-key_states[..., 64:], key_states[..., :64]), dim=-1)
    return c, s, qrot, krot

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
    kv_states = F.linear(
        hidden_states,
        torch.cat((k_weight, v_weight), dim=0),
        torch.cat((k_bias, v_bias), dim=0),
    )
    key_states, value_states = kv_states.split(num_kv_heads * head_dim, dim=-1)
    
    # Reshape to [batch, num_heads, seq_len, head_dim]
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # Compilation removes several memory-bound concatenation kernels, but its
    # fixed launch overhead is not worthwhile for small token batches.
    if bsz * q_len <= 512:
        rope_cos = torch.cat((cos[0, ..., :32], cos[1, ..., 32:80], cos[2, ..., 80:]), dim=-1).unsqueeze(1)
        rope_sin = torch.cat((sin[0, ..., :32], sin[1, ..., 32:80], sin[2, ..., 80:]), dim=-1).unsqueeze(1)
        query_rot = torch.cat((-query_states[..., 64:], query_states[..., :64]), dim=-1)
        key_rot = torch.cat((-key_states[..., 64:], key_states[..., :64]), dim=-1)
    else:
        rope_cos, rope_sin, query_rot, key_rot = _apply_rope(query_states, key_states, cos, sin)
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
