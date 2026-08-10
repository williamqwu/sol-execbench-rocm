import torch
import torch.nn.functional as F

@torch.no_grad()
def run(hidden_states, qkv_weight, qkv_bias, q_norm_weight, q_norm_bias,
        k_norm_weight, k_norm_bias, out_proj_weight, out_proj_bias, eps):
    b, s, d = hidden_states.shape
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias).view(b, s, 3, 24, 64)
    q, k, v = qkv.unbind(2)
    q = F.layer_norm(q, (64,), q_norm_weight, q_norm_bias, eps).transpose(1, 2)
    k = F.layer_norm(k, (64,), k_norm_weight, k_norm_bias, eps).transpose(1, 2)
    v = v.transpose(1, 2)
    x = F.scaled_dot_product_attention(q, k, v)
    x = x.transpose(1, 2).reshape(b, s, d)
    return F.linear(x, out_proj_weight, out_proj_bias)
