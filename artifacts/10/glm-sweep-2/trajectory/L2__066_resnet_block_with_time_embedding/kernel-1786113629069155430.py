import torch
import torch.nn.functional as F


@torch.compile(mode="max-autotune-no-cudagraphs")
def _run(
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
):
    residual = x

    h = F.group_norm(x, num_groups=32, weight=norm1_weight, bias=norm1_bias, eps=norm_eps)
    h = F.silu(h)
    h = F.conv2d(h, conv1_weight, conv1_bias, stride=1, padding=1)

    t = F.silu(time_emb)
    t = F.linear(t, time_emb_proj_weight, time_emb_proj_bias)
    h = h + t[:, :, None, None]

    h = F.group_norm(h, num_groups=32, weight=norm2_weight, bias=norm2_bias, eps=norm_eps)
    h = F.silu(h)
    h = F.conv2d(h, conv2_weight, conv2_bias, stride=1, padding=1)

    return h + residual


@torch.no_grad()
def run(
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
):
    return _run(
        x, time_emb, norm1_weight, norm1_bias, conv1_weight, conv1_bias,
        time_emb_proj_weight, time_emb_proj_bias, norm2_weight, norm2_bias,
        conv2_weight, conv2_bias, norm_eps,
    )
