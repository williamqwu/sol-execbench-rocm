import torch


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    B, S, H = attn_output.shape
    M = B * S
    # For large M the GEMM is compute-bound; fusing the residual add into the
    # matmul epilogue (addmm) saves a full output write+read pass over HBM.
    # For small/medium M launch/dispatch overhead dominates and the 3-D matmul
    # selects a better-tiled kernel; the separate add is cheap there.
    if M >= 6000:
        a2 = attn_output.reshape(M, H)
        r2 = residual.reshape(M, H)
        out = torch.addmm(r2, a2, o_proj_weight.t(), beta=1.0, alpha=1.0)
        return out.view(B, S, H)
    return torch.matmul(attn_output, o_proj_weight.t()) + residual
