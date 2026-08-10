import torch


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous(memory_format=torch.channels_last)
    weight = weight.contiguous(memory_format=torch.channels_last)
    return torch.conv2d(x, weight, bias, stride=2, padding=1)
