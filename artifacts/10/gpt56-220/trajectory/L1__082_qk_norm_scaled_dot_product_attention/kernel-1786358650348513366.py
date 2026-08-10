import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_norm_bias: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    eps: float,
):
    batch_size, seq_len, dim = hidden_states.shape
    num_heads = 24
    head_dim = 64
    scale = head_dim ** -0.5
    
    # QKV projection: (batch, seq_len, dim) -> (batch, seq_len, 3 * dim)
    qkv = torch.addmm(qkv_bias, hidden_states.reshape(-1, dim), qkv_weight.t())
    
    # Reshape and split: (batch, seq_len, 3 * dim) -> 3 x (batch, num_heads, seq_len, head_dim)
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # Apply LayerNorm to Q: normalize across head_dim
    # LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
    q_mean = q.mean(dim=-1, keepdim=True)
    q_var = q.var(dim=-1, unbiased=False, keepdim=True)
    q_normalized = (q - q_mean) / torch.sqrt(q_var + eps)
    # Fold the exact power-of-two attention scale into the affine parameters.
    q = q_normalized * (q_norm_weight * scale) + (q_norm_bias * scale)
    
    # Apply LayerNorm to K: normalize across head_dim
    k_mean = k.mean(dim=-1, keepdim=True)
    k_var = k.var(dim=-1, unbiased=False, keepdim=True)
    k_normalized = (k - k_mean) / torch.sqrt(k_var + eps)
    k = k_normalized * k_norm_weight + k_norm_bias
    
    # Compute attention scores: (batch, num_heads, seq_len, seq_len)
    q_bmm = q.reshape(batch_size * num_heads, seq_len, head_dim)
    k_bmm = k.reshape(batch_size * num_heads, seq_len, head_dim)
    v_bmm = v.reshape(batch_size * num_heads, seq_len, head_dim)
    attn_scores = torch.bmm(q_bmm, k_bmm.transpose(1, 2))
    
    # Softmax over last dimension
    attn_probs = torch.softmax(attn_scores, dim=-1, out=attn_scores)
    
    # Apply attention to values: (batch, num_heads, seq_len, head_dim)
    torch.bmm(attn_probs, v_bmm, out=q_bmm)
    attn_output = q_bmm.reshape(batch_size, num_heads, seq_len, head_dim)
    
    # Reshape: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, dim)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
    
    # Output projection
    output = torch.addmm(
        out_proj_bias, attn_output.reshape(-1, dim), out_proj_weight.t()
    ).reshape(batch_size, seq_len, dim)
    
    return output
