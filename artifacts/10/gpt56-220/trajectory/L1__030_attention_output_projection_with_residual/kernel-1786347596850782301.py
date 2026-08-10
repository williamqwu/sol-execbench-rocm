import torch

torch.backends.cuda.preferred_blas_library("hipblaslt")


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(attn_output, o_proj_weight.t()).add_(residual)
