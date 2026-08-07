import torch
import torch.nn.functional as F

@torch.compile(dynamic=True)
def _fused(x, norm_weight, k_proj_weight, v_proj_weight, eps):
    bs, sl, h = x.shape
    num_kv_heads = 2
    head_dim = 64
    kv_hidden = num_kv_heads * head_dim

    x2d = x.reshape(-1, h)
    xf = x2d.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    normalized = (norm_weight * xf).to(x2d.dtype)

    keys_flat = F.linear(normalized, k_proj_weight, bias=None)
    values_flat = F.linear(normalized, v_proj_weight, bias=None)

    keys = keys_flat.view(bs, sl, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    values = values_flat.view(bs, sl, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    return keys, values


@torch.no_grad()
def run(
    encoder_hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    eps: float,
):
    return _fused(encoder_hidden_states, norm_weight, k_proj_weight, v_proj_weight, eps)
