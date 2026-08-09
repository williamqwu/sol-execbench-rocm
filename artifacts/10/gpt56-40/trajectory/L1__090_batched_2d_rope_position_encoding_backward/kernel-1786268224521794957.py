import torch


@torch.compile
def _rope_backward_complex(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    idx_theta: torch.Tensor,
) -> torch.Tensor:
    phase = torch.exp(torch.complex(torch.zeros_like(idx_theta), idx_theta))
    return grad_sin.float() * phase.real - grad_cos.float() * phase.imag


@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    idx_theta: torch.Tensor,
) -> torch.Tensor:
    return _rope_backward_complex(grad_cos, grad_sin, idx_theta)
