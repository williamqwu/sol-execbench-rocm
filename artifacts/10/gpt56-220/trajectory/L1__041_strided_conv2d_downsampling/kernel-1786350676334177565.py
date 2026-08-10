import torch


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.thnn_conv2d.default(
        x, weight, [3, 3], bias, [2, 2], [1, 1]
    )
