import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    # bf16 GEMM, fp32 accum, fold scaling via baddbmm alpha (no prescale pass).
    B, Hq, S, D = query.shape
    q = query.reshape(B * Hq, S, D)
    k = key.expand(B, Hq, S, D).reshape(B * Hq, S, D)
    kt = k.transpose(1, 2)
    out = torch.empty(B * Hq, S, S, dtype=query.dtype, device=query.device)
    torch.baddbmm(out, q, kt, alpha=scaling, beta=0.0, out=out)
    return out.reshape(B, Hq, S, S)
