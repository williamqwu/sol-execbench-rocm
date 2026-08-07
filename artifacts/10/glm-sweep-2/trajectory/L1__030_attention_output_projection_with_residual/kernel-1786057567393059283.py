import torch


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    B, S, H = attn_output.shape
    a2 = attn_output.reshape(B * S, H)
    r2 = residual.reshape(B * S, H)
    # addmm fuses: out = beta*residual + alpha*(a @ w). Single kernel, no intermediate.
    out = torch.addmm(r2, a2, o_proj_weight.t(), beta=1.0, alpha=1.0)
    return out.view(B, S, H)
