import torch
import torch.nn.functional as F


@torch.jit.script
def _impl(
    x: torch.Tensor,
    time_emb: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    time_emb_proj_weight: torch.Tensor,
    time_emb_proj_bias: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    conv2_bias: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    residual = x
    h = F.group_norm(x, 32, norm1_weight, norm1_bias, norm_eps)
    h.mul_(torch.sigmoid(h))
    h = F.conv2d(h, conv1_weight, conv1_bias, padding=1)
    t = time_emb * torch.sigmoid(time_emb)
    t = F.linear(t, time_emb_proj_weight, time_emb_proj_bias)
    h.add_(t[:, :, None, None])
    h = F.group_norm(h, 32, norm2_weight, norm2_bias, norm_eps)
    h.mul_(torch.sigmoid(h))
    h = F.conv2d(h, conv2_weight, conv2_bias, padding=1)
    h.add_(residual)
    return h


@torch.no_grad()
def run(
    x,
    time_emb,
    norm1_weight,
    norm1_bias,
    conv1_weight,
    conv1_bias,
    time_emb_proj_weight,
    time_emb_proj_bias,
    norm2_weight,
    norm2_bias,
    conv2_weight,
    conv2_bias,
    norm_eps,
):
    return _impl(
        x,
        time_emb,
        norm1_weight,
        norm1_bias,
        conv1_weight,
        conv1_bias,
        time_emb_proj_weight,
        time_emb_proj_bias,
        norm2_weight,
        norm2_bias,
        conv2_weight,
        conv2_bias,
        norm_eps,
    )
