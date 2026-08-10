import torch

torch.backends.cuda.preferred_blas_library("hipblaslt")


def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    hidden = attn_output.shape[-1]
    x = attn_output.view(1, -1, hidden)
    weight_t = o_proj_weight.t().unsqueeze(0)
    projected = torch.bmm(x, weight_t).view_as(attn_output)
    return projected.add_(residual)
