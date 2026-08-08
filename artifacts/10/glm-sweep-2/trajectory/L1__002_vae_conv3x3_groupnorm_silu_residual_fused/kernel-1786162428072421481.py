import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    num_groups = 32
    residual = x
    out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
    out = F.group_norm(out, num_groups, weight=norm1_weight, bias=norm1_bias, eps=eps)
    out = F.silu(out)
    out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
    out = F.group_norm(out, num_groups, weight=norm2_weight, bias=norm2_bias, eps=eps)
    out = F.silu(out)
    out = out + residual
    return out

run = torch.compile(run, mode="max-autotune", fullgraph=True)
