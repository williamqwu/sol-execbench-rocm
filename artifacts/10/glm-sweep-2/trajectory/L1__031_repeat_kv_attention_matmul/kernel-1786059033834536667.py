import torch

@torch.no_grad()
@torch.compile
def run(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    head_dim = 128
    scaling = head_dim ** -0.5
    B, H, S, D = query.shape
    q = query.reshape(B * H, S, D)
    k = key.reshape(B * H, S, D)
    attn = torch.baddbmm(
        torch.empty((), dtype=query.dtype, device=query.device),
        q, k.transpose(1, 2), beta=0.0, alpha=scaling,
    )
    return attn.reshape(B, H, S, S)
