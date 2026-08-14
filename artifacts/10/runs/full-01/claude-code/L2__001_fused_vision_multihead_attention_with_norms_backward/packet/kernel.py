import torch


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    x_mean: torch.Tensor,
    x_var: torch.Tensor,
    x_norm: torch.Tensor,
    ln_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_output: torch.Tensor,
    out_weight: torch.Tensor,
    scale: float,
    norm_eps: float,
):
    B, S, E = x.shape
    H, D = 16, 64
    N = B * S

    go2 = grad_output.reshape(N, E)
    grad_attn_output = go2 @ out_weight
    grad_out_weight = go2.t() @ attn_output.reshape(N, E)
    grad_out_bias = grad_output.sum(dim=(0, 1))

    gaoh = grad_attn_output.view(B, S, H, D).transpose(1, 2)

    grad_v = torch.matmul(attn_weights.transpose(-2, -1), gaoh)
    dA = torch.matmul(gaoh, v.transpose(-2, -1))
    sum_grad = (dA * attn_weights).sum(dim=-1, keepdim=True)
    dS = attn_weights * (dA - sum_grad)
    grad_q = torch.matmul(dS, k) * scale
    grad_k = torch.matmul(dS.transpose(-2, -1), q * scale)

    grad_qkv = torch.empty(B, S, 3 * E, device=x.device, dtype=x.dtype)
    gv = grad_qkv.view(B, S, 3, H, D).permute(0, 2, 3, 1, 4)
    gv[:, 0].copy_(grad_q)
    gv[:, 1].copy_(grad_k)
    gv[:, 2].copy_(grad_v)

    gq2 = grad_qkv.reshape(N, 3 * E)
    grad_x_norm = gq2 @ qkv_weight
    grad_qkv_weight = gq2.t() @ x_norm.reshape(N, E)
    grad_qkv_bias = grad_qkv.sum(dim=(0, 1))

    grad_x_norm = grad_x_norm.view(B, S, E)

    std = torch.sqrt(x_var + norm_eps)
    x_centered = x - x_mean
    x_normalized = x_centered / std

    grad_ln_weight = (grad_x_norm * x_normalized).sum(dim=(0, 1))
    grad_ln_bias = grad_x_norm.sum(dim=(0, 1))

    grad_x_normalized = grad_x_norm * ln_weight

    grad_x_from_norm = grad_x_normalized / std
    mean_grad = grad_x_from_norm.mean(dim=-1, keepdim=True)
    mean_grad_x_centered = (grad_x_normalized * x_centered).mean(
        dim=-1, keepdim=True
    ) / (std ** 2)

    grad_x_from_norm = grad_x_from_norm - mean_grad - x_centered * mean_grad_x_centered

    grad_x = grad_output + grad_x_from_norm

    return (
        grad_x,
        grad_qkv_weight,
        grad_qkv_bias,
        grad_out_weight,
        grad_out_bias,
        grad_ln_weight,
        grad_ln_bias,
    )
