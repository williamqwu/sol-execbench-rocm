import torch


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    shape = attn_output.shape
    x = attn_output.reshape(-1, shape[-1])
    r = residual.reshape(-1, shape[-1])
    return torch.addmm(r, x, o_proj_weight.t()).reshape(shape)
