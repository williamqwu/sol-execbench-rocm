import torch


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.miopen_convolution.default(
        x, weight, bias, [1, 1], [2, 2], [1, 1], 1, True, False
    )
