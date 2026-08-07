import torch
import torch.nn.functional as F

@torch.compile(dynamic=True)
def _fused(x, norm_weight, w_kv, kv_hidden, eps):
    bs, sl, h = x.shape
    num_kv_heads = 2
    head_dim = 64

    x2d = x.reshape(-1, h)
    xf = x2d.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    normalized = (norm_weight * xf).to(x2d.dtype)

    kv_flat = F.linear(normalized, w_kv, bias=None)  # [N, 256]

    keys = kv_flat[:, :kv_hidden].view(bs, sl, num_kv_heads, head_dim).transpose(1, 2)
    values = kv_flat[:, kv_hidden:].view(bs, sl, num_kv_heads, head_dim).transpose(1, 2)
    return keys, values


@torch.no_grad()
def run(
    encoder_hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    eps: float,
):
    kv_hidden = 128
    w_kv = torch.cat([k_proj_weight, v_proj_weight], dim=0)
    return _fused(encoder_hidden_states, norm_weight, w_kv, kv_hidden, eps)
