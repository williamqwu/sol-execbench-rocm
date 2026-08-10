import torch
import torch.nn.functional as F

@torch.compile(fullgraph=True)
def _norm_rope(q, k, cos, sin, q_weight, k_weight, eps):
    qf = q.float()
    kf = k.float()
    q = q_weight * (qf * torch.rsqrt(qf.square().mean(-1, keepdim=True) + eps)).to(q.dtype)
    k = k_weight * (kf * torch.rsqrt(kf.square().mean(-1, keepdim=True) + eps)).to(k.dtype)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    q1, q2 = q[..., :64], q[..., 64:]
    k1, k2 = k[..., :64], k[..., 64:]
    q = torch.cat((q1 * c[..., :64] - q2 * s[..., :64],
                   q2 * c[..., 64:] + q1 * s[..., 64:]), -1)
    k = torch.cat((k1 * c[..., :64] - k2 * s[..., :64],
                   k2 * c[..., 64:] + k1 * s[..., 64:]), -1)
    return q, k

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
    query_states = F.linear(hidden_states, q_proj_weight)
    # K: (batch, seq_len, 4096) @ (1024, 4096).T -> (batch, seq_len, 1024)
    key_states = F.linear(hidden_states, k_proj_weight)
    # V: (batch, seq_len, 4096) @ (1024, 4096).T -> (batch, seq_len, 1024)
    value_states = F.linear(hidden_states, v_proj_weight)
    
    # Reshape to (batch, seq_len, num_heads, head_dim)
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    
    # Per-head RMSNorm on Q and K
    def rms_norm(x, weight, eps):
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + eps)
        return (weight * x_normed.to(input_dtype))
    
    query_states, key_states = _norm_rope(
        query_states, key_states, cos, sin,
        q_norm_weight, k_norm_weight, rms_norm_eps)
    
    # Transpose to (batch, num_heads, seq_len, head_dim)
    # Q and K were transposed by the fused normalization/RoPE kernel.
    value_states = value_states.transpose(1, 2)
    
    # Apply RoPE
    # cos, sin: (batch, seq_len, head_dim) -> (batch, 1, seq_len, head_dim)
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    # RoPE was applied by the fused normalization/RoPE kernel.
    
    # Repeat KV heads for grouped query attention (8 -> 32 heads)
    # key_states: (batch, 8, seq_len, 128) -> (batch, 32, seq_len, 128)
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
    
    # Reshape: (batch, 32, seq_len, 128) -> (batch, seq_len, 32, 128) -> (batch, seq_len, 4096)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    
    # Output projection
    attn_output = F.linear(attn_output, o_proj_weight)
    
    return attn_output
