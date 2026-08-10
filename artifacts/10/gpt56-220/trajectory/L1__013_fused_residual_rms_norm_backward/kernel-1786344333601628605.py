import torch

torch._dynamo.config.recompile_limit = 32
torch._inductor.config.triton.cooperative_reductions = True

@torch.compile(fullgraph=True, dynamic=False)
def _compiled(grad_output: torch.Tensor, x: torch.Tensor,
              normalized: torch.Tensor, rstd: torch.Tensor,
    weight: torch.Tensor):
    g = grad_output
    prod = g * normalized
    grad_weight = prod.reshape(-1, 2560).sum(dim=0)
    gn = g * weight
    mean = (prod * weight).mean(dim=-1, keepdim=True)
    dx = (rstd * (gn - mean * normalized)).to(torch.bfloat16)
    return dx, dx, grad_weight


@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor,
        normalized: torch.Tensor, rstd: torch.Tensor,
        weight: torch.Tensor):
    return _compiled(grad_output, x, normalized, rstd, weight)
