import torch

@torch.no_grad()
@torch.compile(dynamic=True)
def _rope(x, cos, sin, half_dim):
    cos1, cos2 = cos[..., :half_dim], cos[..., half_dim:]
    sin1, sin2 = sin[..., :half_dim], sin[..., half_dim:]
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    out1 = x1 * cos1 - x2 * sin1
    out2 = x2 * cos2 + x1 * sin2
    return torch.cat([out1, out2], dim=-1)

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
    query_rotated = _rope(query, cos, sin, half_dim)
    key_rotated = _rope(key, cos, sin, half_dim)
    return query_rotated, key_rotated
