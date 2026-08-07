import torch
import torch.nn.functional as F

@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    """GELU activation using tanh approximation (fused F.gelu)."""
    return F.gelu(x, approximate='tanh')
