import torch
import torch.nn.functional as F

@torch.no_grad()
def run(hidden_states, qkv_weight, qkv_bias, q_norm_weight, q_norm_bias,
        k_norm_weight, k_norm_bias, out_proj_weight, out_proj_bias, eps):
    b, s, d = hidden_states.shape
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias).view(b, s, 3, 24, 64)
    q, k, v = qkv.unbind(2)
    qm = q.mean(-1, keepdim=True); qv = q.var(-1, unbiased=False, keepdim=True)
    km = k.mean(-1, keepdim=True); kv = k.var(-1, unbiased=False, keepdim=True)
    q = (((q-qm)/torch.sqrt(qv+eps))*q_norm_weight+q_norm_bias).transpose(1, 2)
    k = (((k-km)/torch.sqrt(kv+eps))*k_norm_weight+k_norm_bias).transpose(1, 2)
    v = v.transpose(1, 2)
    x = F.scaled_dot_product_attention(q, k, v)
    x = x.transpose(1, 2).reshape(b, s, d)
    return F.linear(x, out_proj_weight, out_proj_bias)
