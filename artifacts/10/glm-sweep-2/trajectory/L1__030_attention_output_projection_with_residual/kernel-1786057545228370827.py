import torch


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    B, S, H = attn_output.shape
    a2 = attn_output.view(B * S, H)
    # baddbmm: out = beta*residual + alpha*(a @ b). Fuse residual into epilogue.
    r2 = residual.view(B * S, H)
    a2b = a2.unsqueeze(1)        # [BS, 1, H]
    w = o_proj_weight.t().unsqueeze(0)  # [1, H, H]
    r2b = r2.unsqueeze(1)       # [BS, 1, H]
    out = torch.baddbmm(r2b, a2b, w, beta=1.0, alpha=1.0)
    return out.view(B, S, H)
