import torch
import torch.nn.functional as F


@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, weight_q: torch.Tensor,
        weight_k: torch.Tensor, eps: float):
    query_norm = F.rms_norm(query, (128,), weight=None, eps=eps)
    key_norm = F.rms_norm(key, (128,), weight=None, eps=eps)
    return query_norm * weight_q, key_norm * weight_k
