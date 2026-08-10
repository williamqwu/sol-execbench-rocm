import math
import torch


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def _gelu_formula(x: torch.Tensor) -> torch.Tensor:
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + torch.tanh(inner))


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    return _gelu_formula(x)
