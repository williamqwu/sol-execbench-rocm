import torch
import torch.nn.functional as F

@torch.compile(dynamic=True)
def _geglu(x):
    x_gate, x_linear = x.chunk(2, dim=-1)
    return F.gelu(x_gate, approximate='tanh') * x_linear

@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    return _geglu(x)
