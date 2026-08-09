import torch


def _rotate_into(x, cos, sin):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    out = torch.empty_like(x)
    first = out[..., :half]
    second = out[..., half:]
    torch.mul(x1, cos[None, None, :, :half], out=first)
    first.addcmul_(x2, sin[None, None, :, :half], value=-1.0)
    torch.mul(x2, cos[None, None, :, half:], out=second)
    second.addcmul_(x1, sin[None, None, :, half:])
    return out


@torch.no_grad()
def run(query, key, cos, sin):
    return _rotate_into(query, cos, sin), _rotate_into(key, cos, sin)
