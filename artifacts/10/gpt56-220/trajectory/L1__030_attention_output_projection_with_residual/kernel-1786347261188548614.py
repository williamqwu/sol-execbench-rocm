import torch


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    hidden = attn_output.shape[-1]
    projected = torch.mm(attn_output.view(-1, hidden), o_proj_weight.t())
    return projected.view_as(attn_output).add_(residual)
