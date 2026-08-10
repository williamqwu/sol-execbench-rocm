import torch


@torch.compile(fullgraph=True)
def _gelu_formula(x: torch.Tensor) -> torch.Tensor:
    inner = 0.7978845608028654 * x + 0.035677408136300125 * x * x * x
    return 0.5 * x * (1.0 + torch.tanh(inner))


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    return _gelu_formula(x)
