import torch


@torch.compile
def _rope_backward(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    idx_theta: torch.Tensor,
) -> torch.Tensor:
    return (-grad_cos.float() * torch.sin(idx_theta)
            + grad_sin.float() * torch.cos(idx_theta))


@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    idx_theta: torch.Tensor,
) -> torch.Tensor:
    return _rope_backward(grad_cos, grad_sin, idx_theta)
