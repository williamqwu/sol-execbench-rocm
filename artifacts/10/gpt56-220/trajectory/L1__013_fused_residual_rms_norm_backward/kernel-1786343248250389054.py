import torch


@torch.compile(fullgraph=True, dynamic=True)
def _compiled(grad_output: torch.Tensor, x: torch.Tensor,
              normalized: torch.Tensor, rstd: torch.Tensor,
              weight: torch.Tensor):
    g = grad_output.float()
    grad_weight = (g * normalized).sum(dim=(0, 1))
    gn = g * weight
    mean = (gn * normalized).mean(dim=-1, keepdim=True)
    dx = (rstd * (gn - mean * normalized)).to(torch.bfloat16)
    return dx, dx.clone(), grad_weight


@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor,
        normalized: torch.Tensor, rstd: torch.Tensor,
        weight: torch.Tensor):
    return _compiled(grad_output, x, normalized, rstd, weight)
