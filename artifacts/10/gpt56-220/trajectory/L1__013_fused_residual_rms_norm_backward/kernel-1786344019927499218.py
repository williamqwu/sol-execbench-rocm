import torch

torch._dynamo.config.recompile_limit = 32

@torch.compile(fullgraph=True, dynamic=False, mode="reduce-overhead")
def _compiled(grad_output: torch.Tensor, x: torch.Tensor,
              normalized: torch.Tensor, rstd: torch.Tensor,
    weight: torch.Tensor):
    g = grad_output
    g2 = g.reshape(-1, 2560)
    n2 = normalized.reshape(-1, 2560)
    grad_weight = (g2 * n2).sum(dim=0)
    gn = g * weight
    mean = (gn * normalized).mean(dim=-1, keepdim=True)
    dx = (rstd * (gn - mean * normalized)).to(torch.bfloat16)
    return dx, dx, grad_weight


@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor,
        normalized: torch.Tensor, rstd: torch.Tensor,
        weight: torch.Tensor):
    return _compiled(grad_output, x, normalized, rstd, weight)
