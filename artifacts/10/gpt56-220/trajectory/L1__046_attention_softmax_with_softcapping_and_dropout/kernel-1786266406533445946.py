import torch
import torch.nn.functional as F


@torch.no_grad()
def run(attn_weights: torch.Tensor) -> torch.Tensor:
    # Keep eager/vendor softmax, but collapse the softcap chain to one temporary.
    softcapped = attn_weights.clone()
    softcapped.div_(30.0).tanh_().mul_(30.0)
    return F.softmax(softcapped, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
