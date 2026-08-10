import torch

torch._dynamo.config.cache_size_limit = 32
torch._dynamo.config.recompile_limit = 32


@torch.compile(fullgraph=True, dynamic=False, mode="max-autotune")
def _compiled(query, key, weight_q, weight_k, eps):
    q_var = (query * query).mean(dim=-1, keepdim=True)
    k_var = (key * key).mean(dim=-1, keepdim=True)
    q_out = query * torch.rsqrt(q_var + eps) * weight_q
    k_out = key * torch.rsqrt(k_var + eps) * weight_k
    return q_out, k_out


@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, weight_q: torch.Tensor,
        weight_k: torch.Tensor, eps: float):
    return _compiled(query, key, weight_q, weight_k, eps)
