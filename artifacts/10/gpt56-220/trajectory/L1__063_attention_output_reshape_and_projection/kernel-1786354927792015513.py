import torch
import torch.nn.functional as F


@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    x = attn_output.movedim(1, 2).flatten(2)
    return F.linear(x, o_proj_weight)
