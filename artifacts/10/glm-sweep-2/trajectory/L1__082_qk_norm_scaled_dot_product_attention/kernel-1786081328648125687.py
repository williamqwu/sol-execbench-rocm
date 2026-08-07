import torch
import torch.nn.functional as F

@torch.no_grad()
def run(hidden_states, qkv_weight, qkv_bias, q_norm_weight, q_norm_bias, k_norm_weight, k_norm_bias, out_proj_weight, out_proj_bias, eps):
    batch_size, seq_len, dim = hidden_states.shape
    num_heads = 24; head_dim = 64; scale = head_dim ** -0.5
    qkv = torch.matmul(hidden_states, qkv_weight.t()) + qkv_bias
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4).contiguous()
    q, k, v = qkv[0], qkv[1], qkv[2]
    q_mean = q.mean(dim=-1, keepdim=True); q_var = q.var(dim=-1, unbiased=False, keepdim=True)
    q = (q - q_mean) / torch.sqrt(q_var + eps); q = q * q_norm_weight + q_norm_bias
    k_mean = k.mean(dim=-1, keepdim=True); k_var = k.var(dim=-1, unbiased=False, keepdim=True)
    k = (k - k_mean) / torch.sqrt(k_var + eps); k = k * k_norm_weight + k_norm_bias
    attn_output = F.scaled_dot_product_attention(q, k, v)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
    output = torch.matmul(attn_output, out_proj_weight.t()) + out_proj_bias
    return output
