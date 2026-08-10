import torch

@torch.compile(fullgraph=True, mode="reduce-overhead")
def _conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.conv2d(x, weight, bias, stride=2, padding=1)

@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return _conv(x, weight, bias)
