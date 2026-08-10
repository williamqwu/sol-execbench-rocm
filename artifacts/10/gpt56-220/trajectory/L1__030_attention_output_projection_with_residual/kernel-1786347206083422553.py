import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    return F.linear(attn_output, o_proj_weight).add_(residual)
