import torch
import triton
import triton.language as tl


@triton.jit
def _small_kernel(gk, gv, key, cos, sin, pos,
                  out_k, out_v, out_cos, out_sin,
                  S: tl.constexpr, M: tl.constexpr, H: tl.constexpr = 8, D: tl.constexpr = 128,
                  BLOCK: tl.constexpr = 128):
    bs = tl.program_id(0)
    b = bs // S
    s = bs - b * S
    d = tl.arange(0, BLOCK)
    p = tl.load(pos + s)
    c = tl.load(cos + (b * S + s) * D + d)
    sn = tl.load(sin + (b * S + s) * D + d)
    csum = tl.zeros((BLOCK,), tl.float32)
    ssum = tl.zeros((BLOCK,), tl.float32)
    for h in tl.static_range(0, H):
        cache_off = ((b * H + h) * M + p) * D + d
        x = tl.load(gk + cache_off)
        kval = tl.load(key + ((b * H + h) * S + s) * D + d)
        other_d = tl.where(d < 64, d + 64, d - 64)
        other = tl.load(sin + (b * S + s) * D + other_d)
        # d<64: x*c + gradient arriving from second half's rotate term.
        # d>=64: x*c - gradient arriving from first half's rotate term.
        xother = tl.load(gk + ((b * H + h) * M + p) * D + other_d)
        g = x * c + tl.where(d < 64, xother * other, -xother * other)
        tl.store(out_k + ((b * H + h) * S + s) * D + d, g)
        tl.store(out_v + ((b * H + h) * S + s) * D + d, tl.load(gv + cache_off))
        csum += x * kval
        # rotated key is [-k2, k1]
        kother = tl.load(key + ((b * H + h) * S + s) * D + other_d)
        ssum += x * tl.where(d < 64, -kother, kother)
    tl.store(out_cos + (b * S + s) * D + d, csum)
    tl.store(out_sin + (b * S + s) * D + d, ssum)


@triton.jit
def _cache_kernel(gk, gv, ok, ov, n_elements: tl.constexpr,
                  S: tl.constexpr, M: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    token = (offs // 128) % M
    keep = token >= S
    tl.store(ok + offs, tl.where(keep, tl.load(gk + offs, mask=mask), 0.0), mask=mask)
    tl.store(ov + offs, tl.where(keep, tl.load(gv + offs, mask=mask), 0.0), mask=mask)

@triton.jit
def _zero_kernel(ok, ov, pos, S: tl.constexpr, M: tl.constexpr,
                 BLOCK: tl.constexpr = 256):
    pid = tl.program_id(0)
    bs = pid // 4
    chunk = pid % 4
    b = bs // S
    s = bs % S
    x = tl.arange(0, BLOCK)
    h = x // 32
    d = chunk * 32 + x % 32
    p = tl.load(pos + s)
    off = ((b * 8 + h) * M + p) * 128 + d
    tl.store(ok + off, 0.0)
    tl.store(ov + off, 0.0)


@torch.no_grad()
def run(grad_key_cache, grad_value_cache, key_states, cos, sin, cache_position):
    B, H, M, D = grad_key_cache.shape
    S = key_states.shape[2]
    out_k = torch.empty_like(key_states)
    out_v = torch.empty_like(key_states)
    out_cos = torch.empty_like(cos)
    out_sin = torch.empty_like(sin)
    ok = torch.empty_like(grad_key_cache)
    ov = torch.empty_like(grad_value_cache)
    _small_kernel[(B * S,)](
        grad_key_cache, grad_value_cache, key_states, cos, sin, cache_position,
        out_k, out_v, out_cos, out_sin, S=S, M=M)
    n = grad_key_cache.numel()
    _cache_kernel[(triton.cdiv(n, 512),)](
        grad_key_cache, grad_value_cache, ok, ov, n_elements=n, S=S, M=M, BLOCK=512)
    return out_k, out_v, out_cos, out_sin, ok, ov
