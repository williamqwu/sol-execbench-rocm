import torch
import torch.nn.functional as F

@torch.compile
def _project_rotary(hidden_states, cos, sin, qkv_weight, qkv_bias):
    n = hidden_states.shape[0]
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias).reshape(n, 3, 16, 80)
    q, k, v = qkv.unbind(1)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    q1, q2 = q[..., :40], q[..., 40:]
    k1, k2 = k[..., :40], k[..., 40:]
    q = q * c + torch.cat((-q2, q1), -1) * s
    k = k * c + torch.cat((-k2, k1), -1) * s
    return q, k, v

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
):
    """
    Vision attention with variable-length sequences via cu_seqlens.
    
    Args:
        hidden_states: (total_seq_len, hidden_size)
        cu_seqlens: (num_seqs,) cumulative sequence lengths, first element is 0
        cos: (total_seq_len, head_dim) rotary cosine embeddings
        sin: (total_seq_len, head_dim) rotary sine embeddings
        qkv_weight: (3 * hidden_size, hidden_size)
        qkv_bias: (3 * hidden_size,)
        proj_weight: (hidden_size, hidden_size)
        proj_bias: (hidden_size,)
    """
    hidden_size = 1280
    num_heads = 16
    head_dim = 80
    scaling = head_dim ** -0.5
    
    total_seq_len = hidden_states.shape[0]
    device = hidden_states.device
    dtype = hidden_states.dtype
    
    # QKV projection: (total_seq_len, hidden_size) -> (total_seq_len, 3 * hidden_size)
    query_states, key_states, value_states = _project_rotary(
        hidden_states, cos, sin, qkv_weight, qkv_bias)
    
    # Reshape for attention: (total_seq_len, num_heads, head_dim) -> (1, num_heads, total_seq_len, head_dim)
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)
    
    # Compute sequence lengths from cu_seqlens
    num_seqs = cu_seqlens.shape[0]
    
    # Process each sequence separately
    attn_outputs = []
    for i in range(num_seqs - 1):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start
        
        if seq_len == 0:
            continue
        
        # Extract sequence
        q_seq = query_states[:, :, start:end, :]  # (1, num_heads, seq_len, head_dim)
        k_seq = key_states[:, :, start:end, :]
        v_seq = value_states[:, :, start:end, :]
        
        attn_weights = torch.matmul(q_seq, k_seq.transpose(2, 3)) * scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
        attn_out = torch.matmul(attn_weights, v_seq)
        attn_out = attn_out.transpose(1, 2)  # (1, seq_len, num_heads, head_dim)
        attn_outputs.append(attn_out)
    
    # Concatenate outputs: (1, total_seq_len, num_heads, head_dim)
    if len(attn_outputs) > 0:
        attn_output = torch.cat(attn_outputs, dim=1)
    else:
        attn_output = torch.zeros(1, 0, num_heads, head_dim, device=device, dtype=dtype)
    
    # Reshape: (total_seq_len, hidden_size)
    attn_output = attn_output.reshape(total_seq_len, hidden_size).contiguous()
    
    # Output projection
    output = F.linear(attn_output, proj_weight, proj_bias)
    
    return output
