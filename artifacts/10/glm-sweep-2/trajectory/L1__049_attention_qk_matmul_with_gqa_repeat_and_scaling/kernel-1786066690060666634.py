import torch

@torch.no_NOGRAD = None
@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    # bf16 GEMM, fp32 accum, fold scaling via baddbmm alpha (no prescale pass).
    # Q: [B,4,S,256], K: [B,1,S,256]. Flatten heads into batch for baddbmm.
    B, Hq, S, D = query.shape
    Hk = key.shape[1]
    q = query.reshape(B * Hq, S, D)
    # repeat key heads to match (baddbmm needs matching batch). Materialize the
    # repeat only here; it's contiguous and small relative to the GEMM.
    k = key.expand(B, Hq, S, D).reshape(B * Hq, S, D)
    kt = k.transpose(1, 2)
    out = torch.empty(B * Hq, S, S, dtype=torch.float32, device=query.device)
    torch.baddbmm(out, q, kt, alpha=scaling, beta=0.0, out=out)
    return out.reshape(B, Hq, S, S).to(query.dtype)
