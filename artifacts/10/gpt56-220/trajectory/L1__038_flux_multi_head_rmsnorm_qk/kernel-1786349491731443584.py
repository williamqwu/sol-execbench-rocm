import torch

torch._dynamo.config.cache_size_limit = 32
torch._dynamo.config.recompile_limit = 32


@torch.compile(fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")
def _compiled(query, key, weight_q, weight_k, eps):
    values = torch.stack((query, key), dim=0)
    weights = torch.stack((weight_q, weight_k), dim=0)
    variance = (values * values).mean(dim=-1, keepdim=True)
    output = values * torch.rsqrt(variance + eps) * weights[:, None, None]
    return output[0], output[1]


@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, weight_q: torch.Tensor,
        weight_k: torch.Tensor, eps: float):
    return _compiled(query, key, weight_q, weight_k, eps)
