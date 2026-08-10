import torch

torch.backends.cuda.preferred_blas_library("hipblaslt")


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    projected = torch.ops.aten.matmul.default(attn_output, o_proj_weight.t())
    return torch.ops.aten.add_.Tensor(projected, residual)
