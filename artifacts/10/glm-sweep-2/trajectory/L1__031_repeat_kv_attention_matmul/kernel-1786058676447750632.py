import torch

_BUF = None

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    global _BUF
    head_dim = 128
    scaling = head_dim ** -0.5
    B, H, S, D = query.shape
    q = query.reshape(B * H, S, D)
    k = key.reshape(B * H, S, D)
    if _BUF is None or _BUF.dtype != query.dtype or _BUF.device != query.device:
        _BUF = torch.empty((), dtype=query.dtype, device=query.device)
    attn = torch.baddbmm(_BUF, q, k.transpose(1, 2), beta=0.0, alpha=scaling)
    return attn.reshape(B, H, S, S)
