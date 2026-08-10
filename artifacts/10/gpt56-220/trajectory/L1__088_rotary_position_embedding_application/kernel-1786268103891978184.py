import torch


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def _rotate(x, cos, sin):
    half = x.shape[-1] // 2
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * c[..., :half] - x2 * s[..., :half],
                      x2 * c[..., half:] + x1 * s[..., half:]), dim=-1)


@torch.no_grad()
def run(query, key, cos, sin):
    return _rotate(query, cos, sin), _rotate(key, cos, sin)
