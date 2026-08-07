"""RoPE (rotate_half form) for MI355X / gfx950.

    out[..., :h] = x[..., :h] * cos[..., :h] - x[..., h:] * sin[..., :h]
    out[..., h:] = x[..., h:] * cos[..., h:] + x[..., :h] * sin[..., h:]

This is a pure streaming problem: every element of q and k is read once,
multiplied twice, and written once.  cos/sin are (seq_len, head_dim) and are
shared by all (batch, head) pairs, so they stay resident in cache.

Two things decide the runtime:

1. **Bandwidth.**  A single kernel handles both q and k, and each program
   processes G heads that share one seq-tile, so the cos/sin tile is fetched
   once per G heads instead of once per head.

2. **Launch overhead.**  The GPU work for the smaller workloads here is only
   3-8 us, while a stock Triton JIT dispatch costs ~9-14 us of pure Python.
   After the first call for a given shape the compiled kernel is invoked
   through its own launcher, which skips the binder/specializer/grid-closure
   machinery.  The cache key reproduces Triton's own specialization rules
   (see `_spec`) so a cached kernel is only reused when Triton itself would
   have reused it.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------


@triton.jit
def _rope(Q, K, QO, KO, COS, SIN, S, H, KVH, NG_Q, HGQ, HGK,
          ROWS: tl.constexpr, HALF: tl.constexpr,
          GQ: tl.constexpr, GK: tl.constexpr):
    pid = tl.program_id(0)
    pid_s = tl.program_id(1)
    D: tl.constexpr = 2 * HALF

    s = pid_s * ROWS + tl.arange(0, ROWS)
    m = (s < S)[:, None]
    col = tl.arange(0, HALF)

    # cos/sin tile: shared by the G heads this program owns.
    coff = s[:, None] * D + col[None, :]
    c1 = tl.load(COS + coff, mask=m, other=0.)
    c2 = tl.load(COS + coff + HALF, mask=m, other=0.)
    n1 = tl.load(SIN + coff, mask=m, other=0.)
    n2 = tl.load(SIN + coff + HALF, mask=m, other=0.)

    if pid < NG_Q:
        b = pid // HGQ
        h0 = (pid % HGQ) * GQ
        off = ((b * H + h0) * S + s)[:, None] * D + col[None, :]
        for i in tl.static_range(GQ):
            if h0 + i < H:
                o = off + i * S * D
                x1 = tl.load(Q + o, mask=m, other=0.)
                x2 = tl.load(Q + o + HALF, mask=m, other=0.)
                tl.store(QO + o, x1 * c1 - x2 * n1, mask=m)
                tl.store(QO + o + HALF, x2 * c2 + x1 * n2, mask=m)
    else:
        p = pid - NG_Q
        b = p // HGK
        h0 = (p % HGK) * GK
        off = ((b * KVH + h0) * S + s)[:, None] * D + col[None, :]
        for i in tl.static_range(GK):
            if h0 + i < KVH:
                o = off + i * S * D
                x1 = tl.load(K + o, mask=m, other=0.)
                x2 = tl.load(K + o + HALF, mask=m, other=0.)
                tl.store(KO + o, x1 * c1 - x2 * n1, mask=m)
                tl.store(KO + o + HALF, x2 * c2 + x1 * n2, mask=m)


# ---------------------------------------------------------------------------
# launch configuration
# ---------------------------------------------------------------------------

_CU = 256          # MI355X compute units
_MIN_PROGRAMS = 8 * _CU


def _group(n):
    for g in (8, 4, 2):
        if n % g == 0:
            return g
    return 1


def _plan(B, H, KV, S):
    gq, gk = _group(H), _group(KV)
    nhg = B * (H // gq + KV // gk)
    rows = 4
    for r in (64, 32, 16, 8, 4):
        if nhg * triton.cdiv(S, r) >= _MIN_PROGRAMS:
            rows = r
            break
    warps = 4 if rows >= 16 else 2
    return gq, gk, rows, warps


# ---------------------------------------------------------------------------
# fast launch path
# ---------------------------------------------------------------------------

try:
    from torch._C import _cuda_getCurrentRawStream as _raw_stream
except Exception:                                          # pragma: no cover
    _raw_stream = None


def _spec(v):
    """Replicate Triton's scalar specialization classes.

    Triton compiles a distinct kernel depending on whether an int argument is
    1 (folded to a constant), divisible by 16 (assumed aligned), or neither.
    Reusing a cached kernel across these classes would miscompile, so the
    class is part of the cache key.
    """
    if v == 1:
        return 1
    if v % 16 == 0:
        return 16
    return 0


_CACHE = {}


def _build(q, k, cos, sin, qo, ko, B, H, KV, S, half):
    gq, gk, rows, warps = _plan(B, H, KV, S)
    hgq = triton.cdiv(H, gq)
    hgk = triton.cdiv(KV, gk)
    ngq = B * hgq
    grid = (ngq + B * hgk, triton.cdiv(S, rows))

    ck = _rope.warmup(q, k, qo, ko, cos, sin, S, H, KV, ngq, hgq, hgk,
                      ROWS=rows, HALF=half, GQ=gq, GK=gk,
                      num_warps=warps, num_stages=1, grid=grid)
    ck._init_handles()
    return (ck, ck.run, ck.function, ck.packed_metadata, grid[0], grid[1],
            rows, gq, gk, ngq, hgq, hgk)


def _torch_fallback(query, key, cos, sin):
    half = query.shape[-1] // 2
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rot(x):
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    return query * cos + rot(query) * sin, key * cos + rot(key) * sin


@torch.no_grad()
def run(query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor):
    B, H, S, D = query.shape
    KV = key.shape[1]
    half = D >> 1

    if (_raw_stream is None or query.dtype is not torch.float32
            or not query.is_cuda or (half & (half - 1)) != 0 or half < 16
            or not (query.is_contiguous() and key.is_contiguous()
                    and cos.is_contiguous() and sin.is_contiguous())
            or key.dtype is not torch.float32
            or cos.dtype is not torch.float32
            or sin.dtype is not torch.float32
            # the cached kernel is compiled assuming 16B-aligned pointers
            or ((query.data_ptr() | key.data_ptr()
                 | cos.data_ptr() | sin.data_ptr()) & 15)):
        return _torch_fallback(query, key, cos, sin)

    q_out = torch.empty_like(query)
    k_out = torch.empty_like(key)

    ck_key = (B, H, KV, S, half)
    ent = _CACHE.get(ck_key)
    if ent is None:
        ent = _build(query, key, cos, sin, q_out, k_out, B, H, KV, S, half)
        # Guard: only reuse this compiled kernel for scalar-argument
        # specialization classes it was actually compiled for.
        ent = ent + ((_spec(S), _spec(H), _spec(KV), _spec(ent[9]),
                      _spec(ent[10]), _spec(ent[11])),)
        _CACHE[ck_key] = ent

    (_, krun, kfn, kmeta, gx, gy, rows, gq, gk, ngq, hgq, hgk, _) = ent

    krun(gx, gy, 1, _raw_stream(query.device.index), kfn, kmeta, None,
         None, None,
         query, key, q_out, k_out, cos, sin,
         S, H, KV, ngq, hgq, hgk,
         rows, half, gq, gk)
    return q_out, k_out
