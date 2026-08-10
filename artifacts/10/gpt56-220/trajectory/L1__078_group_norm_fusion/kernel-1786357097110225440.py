import torch
import torch.nn.functional as F


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    return F.group_norm(x, 32, weight, bias, eps)
