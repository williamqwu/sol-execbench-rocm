import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    scaling: float,
):
    # Constants
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = 4
    
    batch_size, seq_len, _ = hidden_states.shape
    
    # QKV Projections
    # Q: (batch, seq_len, 4096) @ (4096, 4096).T -> (batch, seq_len, 4096)
    if batch_size * seq_len >= 8192:
        query_states = F.linear(hidden_states, q_proj_weight)
        kv_states = F.linear(
            hidden_states, torch.cat((k_proj_weight, v_proj_weight), dim=0))
        key_states, value_states = kv_states.split(1024, dim=-1)
    else:
        query_states = F.linear(hidden_states, q_proj_weight)
        key_states = F.linear(hidden_states, k_proj_weight)
        value_states = F.linear(hidden_states, v_proj_weight)
    
    # Reshape to (batch, seq_len, num_heads, head_dim)
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    # Per-head RMSNorm on Q and K
    def rms_norm(x, weight, eps):
        return F.rms_norm(x, (head_dim,), weight, eps)
    
    query_states = rms_norm(query_states, q_norm_weight, rms_norm_eps)
    key_states = rms_norm(key_states, k_norm_weight, rms_norm_eps)
    
    # Transpose to (batch, num_heads, seq_len, head_dim)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # Apply RoPE
    # cos, sin: (batch, seq_len, head_dim) -> (batch, 1, seq_len, head_dim)
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    query_states = (query_states * cos_expanded) + (rotate_half(query_states) * sin_expanded)
    key_states = (key_states * cos_expanded) + (rotate_half(key_states) * sin_expanded)
    
    # Repeat KV heads for grouped query attention (8 -> 32 heads)
    # key_states: (batch, 8, seq_len, 128) -> (batch, 32, seq_len, 128)
    key_states = torch.repeat_interleave(
        key_states, num_key_value_groups, dim=1)
    value_states = torch.repeat_interleave(
        value_states, num_key_value_groups, dim=1)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    attn_weights = torch.add(attention_mask, attn_weights, alpha=scaling)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Reshape: (batch, 32, seq_len, 128) -> (batch, seq_len, 32, 128) -> (batch, seq_len, 4096)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    
    # Output projection
    attn_output = F.linear(attn_output, o_proj_weight)
    
    return attn_output
