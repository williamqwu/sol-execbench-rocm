import torch


@torch.compile(fullgraph=True)
def _gelu_formula(x: torch.Tensor) -> torch.Tensor:
    inner = x * (0.7978845608028654 + 0.035677408136300125 * x * x)
    return x / (1.0 + torch.exp(-2.0 * inner))


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    return _gelu_formula(x)
