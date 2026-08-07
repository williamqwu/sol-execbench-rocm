import torch

@torch.no_grad()
@torch.compile(dynamic=True)
def _rope_inplace(x, cos, sin, half_dim, out):
    cos1, cos2 = cos[..., :half_dim], cos[..., half_dim:]
    sin1, sin2 = sin[..., :half_dim], sin[..., half_dim:]
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    out[..., :half_dim] = x1 * cos1 - x2 * sin1
    out[..., half_dim:] = x2 * cos2 + x1 * sin2
    return out

@torch.no_grad()
def run(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    half_dim = query.shape[-1] // 2
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    query_rotated = torch.empty_like(query)
    key_rotated = torch.empty_like(key)
    _rope_inplace(query, cos, sin, half_dim, query_rotated)
    _rope_inplace(key, cos, sin, half_dim, key_rotated)
    return query_rotated, key_rotated
