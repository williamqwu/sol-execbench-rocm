import torch
import torch.nn.functional as F


@torch.compile(fullgraph=True)
def _compiled(attn_weights: torch.Tensor) -> torch.Tensor:
    softcapped = torch.tanh(attn_weights / 30.0) * 30.0
    return F.softmax(softcapped, dim=-1, dtype=torch.float32).to(attn_weights.dtype)


@torch.no_grad()
def run(attn_weights: torch.Tensor) -> torch.Tensor:
    return _compiled(attn_weights)
