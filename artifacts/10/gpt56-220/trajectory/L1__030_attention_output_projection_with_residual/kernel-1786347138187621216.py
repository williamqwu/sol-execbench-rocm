import torch


@torch.compile(fullgraph=True, dynamic=True)
def _run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(attn_output, o_proj_weight.t()) + residual


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    return _run(attn_output, residual, o_proj_weight)
