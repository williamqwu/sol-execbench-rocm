import torch
import triton
import triton.language as tl


@triton.jit
def _rope(k, cos, sin, out, total, new_len: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    d = offs & 127
    token = offs // 128
    bh = token // new_len
    nt = token % new_len
    other_d = tl.where(d < 64, d + 64, d - 64)
    x = tl.load(k + offs, mask=mask)
    xr = tl.load(k + token * 128 + other_d, mask=mask)
    xr = tl.where(d < 64, -xr, xr)
    b = bh // 10
    ri = (b * new_len + nt) * 128 + d
    c = tl.load(cos + ri, mask=mask)
    s = tl.load(sin + ri, mask=mask)
    tl.store(out + offs, x * c + xr * s, mask=mask)


@torch.no_grad()
def run(key_states, value_states, cos, sin, key_cache, value_cache):
    rotated = torch.empty_like(key_states)
    count = key_states.numel()
    _rope[(triton.cdiv(count, 256),)](key_states, cos, sin, rotated, count,
                                     key_states.shape[2], BLOCK=256)
    return (torch.cat((key_cache, rotated), dim=2),
            torch.cat((value_cache, value_states), dim=2))
