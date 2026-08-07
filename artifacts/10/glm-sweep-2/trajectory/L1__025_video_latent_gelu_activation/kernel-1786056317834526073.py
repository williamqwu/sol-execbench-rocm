import torch
import math

@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    """GELU tanh via torch.compile fusion."""
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    coeff = 0.044715
    inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    return 0.5 * x * (1.0 + torch.tanh(inner))

run = torch.compile(run, dynamic=True, mode="reduce-overhead")
