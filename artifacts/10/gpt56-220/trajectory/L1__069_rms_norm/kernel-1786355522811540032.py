import torch
import torch.nn.functional as F


@torch.no_grad()
def run(hidden_states: torch.Tensor, residual: torch.Tensor,
        weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = residual + hidden_states
    return F.rms_norm(x, (8192,), weight, eps)
