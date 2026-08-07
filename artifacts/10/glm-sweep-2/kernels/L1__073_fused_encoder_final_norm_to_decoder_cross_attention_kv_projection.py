import torch
import torch.nn.functional as F
from torch._inductor.config import max_autotune_pointwise

max_autotune_pointwise = True

@torch.compile(dynamic=True)
def _fused(x, norm_weight, k_proj_weight, v_proj_weight, eps):
    bs, sl, h = x.shape
    num_kv_heads = 2
    head_dim = 64
    kv_hidden = num_kv_heads * head_dim

    x2d = x.reshape(-1, h)
    w_kv = torch.cat([k_proj_weight, v_proj_weight], dim=0)  # raw, no fold

    xf = x2d.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(var + eps)

    # fold norm_weight into x (broadcast multiply), fed to GEMM
    xn = norm_weight[None, :].to(torch.float32) * xf  # fp32 [N,1024]
    kv_flat = ((xn @ w_kv.to(torch.float32).t()) * inv_rms).to(x2d.dtype)

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
    return _fused(encoder_hidden_states, norm_weight, k_proj_weight, v_proj_weight, eps)
