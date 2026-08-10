import torch


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.conv2d.default(x, weight, bias, [2, 2], [1, 1])
