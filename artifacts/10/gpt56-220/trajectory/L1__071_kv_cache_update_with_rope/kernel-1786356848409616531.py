import torch
import triton
import triton.language as tl


@triton.jit
def _update(k, v, cos, sin, kc, vc, ok, ov,
            n_elements: tl.constexpr, new_len: tl.constexpr,
            old_len: tl.constexpr, total_len: tl.constexpr,
            BLOCK: tl.constexpr):
    local = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = local < n_elements
    bh = tl.program_id(1)
    offs = bh * n_elements + local
    d = local & 127
    t = local // 128
    old = t < old_len
    cache_idx = (bh * old_len + t) * 128 + d
    new_t = t - old_len
    new_idx = (bh * new_len + new_t) * 128 + d

    kval = tl.load(kc + cache_idx, mask=mask & old, other=0.0)
    vval = tl.load(vc + cache_idx, mask=mask & old, other=0.0)
    append = mask & ~old
    x = tl.load(k + new_idx, mask=append, other=0.0)
    other_d = tl.where(d < 64, d + 64, d - 64)
    other_idx = (bh * new_len + new_t) * 128 + other_d
    xr = tl.load(k + other_idx, mask=append, other=0.0)
    xr = tl.where(d < 64, -xr, xr)
    b = bh // 10
    rope_idx = (b * new_len + new_t) * 128 + d
    c = tl.load(cos + rope_idx, mask=append, other=0.0)
    s = tl.load(sin + rope_idx, mask=append, other=0.0)
    rotated = x * c + xr * s
    kval = tl.where(old, kval, rotated)
    vnew = tl.load(v + new_idx, mask=append, other=0.0)
    vval = tl.where(old, vval, vnew)
    tl.store(ok + offs, kval, mask=mask)
    tl.store(ov + offs, vval, mask=mask)


@torch.no_grad()
def run(key_states, value_states, cos, sin, key_cache, value_cache):
    b, h, n, d = key_states.shape
    old = key_cache.shape[2]
    total = old + n
    out_shape = (b, h, total, d)
    out_k = torch.empty(out_shape, device=key_states.device, dtype=key_states.dtype)
    out_v = torch.empty_like(out_k)
    elements_per_head = total * d
    _update[(triton.cdiv(elements_per_head, 1024), b * h)](
        key_states, value_states, cos, sin, key_cache, value_cache, out_k, out_v,
        elements_per_head, n, old, total, BLOCK=1024, num_stages=1)
    return out_k, out_v
