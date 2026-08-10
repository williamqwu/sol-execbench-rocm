import torch
import torch.nn.functional as F


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    x_gate, x_linear = x.chunk(2, dim=-1)
    return F.gelu(x_gate, approximate='tanh') * x_linear
