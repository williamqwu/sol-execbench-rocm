import torch
import torch.nn.functional as F


@torch.no_grad()
def run(sublayer_output: torch.Tensor, residual: torch.Tensor,
        weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = F.rms_norm(sublayer_output, (4096,), weight, eps)
    return residual + normalized
