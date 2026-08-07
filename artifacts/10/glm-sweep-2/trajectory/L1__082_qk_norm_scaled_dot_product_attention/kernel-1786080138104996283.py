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
    
    # QKV projection: (batch, seq_len, dim) -> (batch, seq_len, 3 * dim)
    qkv = torch.matmul(hidden_states, qkv_weight.t()) + qkv_bias
    
    # Reshape and split: (batch, seq_len, 3, num_heads, head_dim)
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # (3, batch, num_heads, seq_len, head_dim)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # Fused LayerNorm across head_dim
    q = F.layer_norm(q, (head_dim,), q_norm_weight, q_norm_bias, eps)
    k = F.layer_norm(k, (head_dim,), k_norm_weight, k_norm_bias, eps)
    
    # Scaled dot-product attention via flash attention.
    # Reference scale = head_dim ** -0.5 == SDPA default, so no scale needed.
    attn_output = F.scaled_dot_product_attention(q, k, v)
    
    # Reshape: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, dim)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
    
    # Output projection
    output = torch.matmul(attn_output, out_proj_weight.t()) + out_proj_bias
    
    return output
