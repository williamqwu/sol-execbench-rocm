import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    head_dim = 128
    scaling = head_dim ** -0.5
    # Fold scaling into the GEMM epilogue via baddbmm alpha; beta=0 ignores input.
    # Reshape [B,H,S,D]->[B*H,S,D] is a free view (contiguous); baddbmm is 3D.
    B, H, S, D = query.shape
    q = query.reshape(B * H, S, D)
    k = key.reshape(B * H, S, D)
    attn = torch.baddbmm(
        torch.empty((), dtype=query.dtype, device=query.device),
        q, k.transpose(1, 2), beta=0.0, alpha=scaling,
    )
    return attn.reshape(B, H, S, S)
