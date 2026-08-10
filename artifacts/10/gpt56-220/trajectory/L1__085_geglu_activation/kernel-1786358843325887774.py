import torch
import torch.nn.functional as F


@torch.compile(fullgraph=True)
def _compiled(x: torch.Tensor) -> torch.Tensor:
    gate, linear = x.chunk(2, dim=-1)
    return F.gelu(gate, approximate="tanh") * linear


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    return _compiled(x)
