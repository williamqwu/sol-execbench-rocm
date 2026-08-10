import torch

@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.pad(x, (1, 1, 1, 1))
    return torch.conv2d(x, weight, bias, stride=2, padding=0)
