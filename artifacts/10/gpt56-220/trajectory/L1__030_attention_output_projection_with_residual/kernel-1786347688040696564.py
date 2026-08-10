import torch

torch.backends.cuda.preferred_blas_library("hipblaslt")


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    batch = attn_output.shape[0]
    weight_t = o_proj_weight.t().unsqueeze(0).expand(batch, -1, -1)
    projected = torch.bmm(attn_output, weight_t)
    return projected.add_(residual)
