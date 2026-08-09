import torch


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def _rotate(x, cos, sin):
    half = x.shape[-1] // 2
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * c + rotated * s


@torch.no_grad()
def run(query, key, cos, sin):
    return _rotate(query, cos, sin), _rotate(key, cos, sin)
