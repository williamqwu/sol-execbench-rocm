import torch

@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    idx_theta: torch.Tensor,
) -> torch.Tensor:
    sin_theta = torch.sin(idx_theta)
    cos_theta = torch.cos(idx_theta)
    grad_idx_theta = -grad_cos.to(torch.float32) * sin_theta + grad_sin.to(torch.float32) * cos_theta
    return grad_idx_theta
