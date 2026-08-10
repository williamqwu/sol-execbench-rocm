import torch


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Direct native binding keeps MIOpen's numerically validated FP32 path.
    return torch.ops.aten._convolution.default(
        x, weight, bias, [2, 2], [1, 1], [1, 1], False, [0, 0], 1,
        False, True, True, True,
    )
