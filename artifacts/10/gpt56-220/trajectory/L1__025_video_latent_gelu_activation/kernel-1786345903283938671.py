import torch
import torch.nn.functional as F


@torch.compile(fullgraph=True, mode="reduce-overhead")
def _gelu_formula(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    return _gelu_formula(x)
