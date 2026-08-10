import torch
import torch.nn.functional as F


@torch.compile
def _rotary(q, k, cos, sin):
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    q1, q2 = q[..., :40], q[..., 40:]
    k1, k2 = k[..., :40], k[..., 40:]
    qr = torch.cat((-q2, q1), dim=-1)
    kr = torch.cat((-k2, k1), dim=-1)
    return q * c + qr * s, k * c + kr * s

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
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    
    # Reshape to (total_seq_len, 3, num_heads, head_dim)
    qkv = qkv.reshape(total_seq_len, 3, num_heads, head_dim)
    qkv = qkv.permute(1, 0, 2, 3)  # (3, total_seq_len, num_heads, head_dim)
    query_states, key_states, value_states = qkv.unbind(0)
    
    # Apply rotary position embeddings
    # query_states, key_states: (total_seq_len, num_heads, head_dim)
    # cos, sin: (total_seq_len, head_dim)
    
    q_float = query_states.float()
    k_float = key_states.float()
    cos_expanded = cos.unsqueeze(1).float()  # (total_seq_len, 1, head_dim)
    sin_expanded = sin.unsqueeze(1).float()  # (total_seq_len, 1, head_dim)
    
    query_states, key_states = _rotary(q_float, k_float, cos, sin)
    
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
        
        # Compute attention scores
        attn_weights = torch.matmul(q_seq, k_seq.transpose(2, 3)) * scaling
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
        
        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v_seq)  # (1, num_heads, seq_len, head_dim)
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
