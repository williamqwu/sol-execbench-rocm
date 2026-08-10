import torch


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Direct native binding keeps MIOpen's numerically validated FP32 path.
    return torch.conv2d(x, weight, bias, stride=2, padding=1)
