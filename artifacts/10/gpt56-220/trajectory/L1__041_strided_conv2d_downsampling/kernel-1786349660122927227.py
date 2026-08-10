import torch


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.convolution.default(
        x, weight, bias, [2, 2], [1, 1], [1, 1], False, [0, 0], 1
    )
