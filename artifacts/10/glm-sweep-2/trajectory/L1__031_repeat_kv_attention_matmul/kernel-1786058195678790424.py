import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    head_dim = 128
    scaling = head_dim ** -0.5
    B, H, S, D = query.shape
    q = query.reshape(B * H, S, D)
    k = key.reshape(B * H, S, D)
    # bf16 output buffer so the GEMM writes bf16 directly (no fp32->bf16 cast).
    out = torch.empty(B * H, S, S, dtype=torch.bfloat16, device=query.device)
    torch.baddbmm(out, q, k.transpose(1, 2), beta=0.0, alpha=scaling, out=out)
    return out.reshape(B, H, S, S)
