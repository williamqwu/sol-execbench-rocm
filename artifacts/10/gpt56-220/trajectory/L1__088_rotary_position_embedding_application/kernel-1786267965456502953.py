import torch


@torch.compile(fullgraph=True, mode="reduce-overhead")
def _rope(query, key, cos, sin):
    half = query.shape[-1] // 2
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]

    q1, q2 = query[..., :half], query[..., half:]
    k1, k2 = key[..., :half], key[..., half:]
    query_rotated = torch.cat((q1 * c[..., :half] - q2 * s[..., :half],
                               q2 * c[..., half:] + q1 * s[..., half:]), dim=-1)
    key_rotated = torch.cat((k1 * c[..., :half] - k2 * s[..., :half],
                             k2 * c[..., half:] + k1 * s[..., half:]), dim=-1)
    return query_rotated, key_rotated


@torch.no_grad()
def run(query, key, cos, sin):
    return _rope(query, key, cos, sin)
