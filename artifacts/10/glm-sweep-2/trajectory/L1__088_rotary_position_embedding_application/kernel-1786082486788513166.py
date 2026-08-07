import torch

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

    cos1, cos2 = cos[..., :half_dim], cos[..., half_dim:]
    sin1, sin2 = sin[..., :half_dim], sin[..., half_dim:]

    q1, q2 = query[..., :half_dim], query[..., half_dim:]
    k1, k2 = key[..., :half_dim], key[..., half_dim:]

    query_rotated = torch.empty_like(query)
    key_rotated = torch.empty_like(key)

    query_rotated[..., :half_dim] = q1 * cos1 - q2 * sin1
    query_rotated[..., half_dim:] = q2 * cos2 + q1 * sin2
    key_rotated[..., :half_dim] = k1 * cos1 - k2 * sin1
    key_rotated[..., half_dim:] = k2 * cos2 + k1 * sin2

    return query_rotated, key_rotated
