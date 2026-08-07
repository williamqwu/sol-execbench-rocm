import torch

@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    """GELU tanh: single fused in-place expression, no F.gelu."""
    x2 = x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x * x2)
    return x.mul_(0.5).mul_(1.0 + torch.tanh(inner))
