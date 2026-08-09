import torch


def _rotate(x, cos, sin):
    shape = x.shape
    flat = x.reshape(-1, shape[-2], shape[-1])
    half = shape[-1] // 2
    c = cos[None, :, :]
    s = sin[None, :, :]
    x1, x2 = flat[..., :half], flat[..., half:]
    first = x1 * c[..., :half] - x2 * s[..., :half]
    second = x2 * c[..., half:] + x1 * s[..., half:]
    out = torch.stack((first, second), dim=-2).reshape(flat.shape)
    return out.reshape(shape)


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def _rope(query, key, cos, sin):
    return _rotate(query, cos, sin), _rotate(key, cos, sin)


@torch.no_grad()
def run(query, key, cos, sin):
    return _rope(query, key, cos, sin)
