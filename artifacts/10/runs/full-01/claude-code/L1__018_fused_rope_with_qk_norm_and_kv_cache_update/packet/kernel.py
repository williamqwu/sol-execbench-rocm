import os
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused RMS-norm(Q,K) + RoPE + KV-cache scatter.
#
# The reference performs, per 128-wide head row:
#     n   = bf16( w.f32 * x.f32 * rsqrt(mean(x^2) + eps) )
#     out_lo = bf16( bf16(n_lo*cos) - bf16(n_hi*sin) )
#     out_hi = bf16( bf16(n_hi*cos) + bf16(n_lo*sin) )
# with cos/sin themselves rounded to bf16 before use.  Every one of those
# intermediate roundings is reproduced below; being *more* accurate than the
# reference fails the tolerance just as surely as being less.
# ---------------------------------------------------------------------------


@triton.jit
def _norm_rope(xlo, xhi, wlo, whi, c, s, eps):
    v = tl.sum(xlo * xlo, 1) + tl.sum(xhi * xhi, 1)
    r = tl.rsqrt(v * 0.0078125 + eps)              # 1/128
    nlo = (xlo * r[:, None] * wlo[None, :]).to(tl.bfloat16).to(tl.float32)
    nhi = (xhi * r[:, None] * whi[None, :]).to(tl.bfloat16).to(tl.float32)
    p1 = (nlo * c).to(tl.bfloat16).to(tl.float32)
    p2 = (nhi * s).to(tl.bfloat16).to(tl.float32)
    p3 = (nhi * c).to(tl.bfloat16).to(tl.float32)
    p4 = (nlo * s).to(tl.bfloat16).to(tl.float32)
    return (p1 - p2).to(tl.bfloat16), (p3 + p4).to(tl.bfloat16)


@triton.jit
def _fused_kernel(
    q_ptr, k_ptr, v_ptr, pos_ptr, kc_ptr, vc_ptr, cp_ptr,
    qw_ptr, kw_ptr, invf_ptr, qo_ptr, ko_ptr,
    eps,
    S, num_sb, NCH, NQC,
    sq0, sq1, sq2,
    sk0, sk1, sk2,
    sv0, sv1, sv2,
    skc0, skc1, skc2,
    svc0, svc1, svc2,
    sp0, sp1,
    HQ: tl.constexpr, HK: tl.constexpr, BS: tl.constexpr,
    EXACT: tl.constexpr, NS: tl.constexpr,
):
    pid = tl.program_id(0)
    sbi = pid % num_sb
    t = pid // num_sb
    hc = t % NCH
    b = t // NCH

    offs_s = sbi * BS + tl.arange(0, BS)
    d = tl.arange(0, 64)
    if EXACT:
        m2 = tl.full((BS, 1), 1, tl.int1)
        pos = tl.load(pos_ptr + b * sp0 + offs_s * sp1).to(tl.float32)
    else:
        m = offs_s < S
        m2 = m[:, None]
        pos = tl.load(pos_ptr + b * sp0 + offs_s * sp1, mask=m, other=0).to(tl.float32)

    # cos / sin for this (batch, seq-block) -- shared by every head in the chunk
    invf = tl.load(invf_ptr + d)
    fr = pos[:, None] * invf[None, :]
    c = tl.cos(fr).to(tl.bfloat16).to(tl.float32)
    s = tl.sin(fr).to(tl.bfloat16).to(tl.float32)

    if hc < NQC:
        # ------------------------------------------------------ query heads
        wlo = tl.load(qw_ptr + d).to(tl.float32)
        whi = tl.load(qw_ptr + 64 + d).to(tl.float32)
        base = b * sq0 + hc * (HQ * sq1) + offs_s[:, None] * sq2 + d[None, :]
        for i in tl.range(0, HQ, num_stages=NS):
            off = base + i * sq1
            xlo = tl.load(q_ptr + off, mask=m2, other=0.0).to(tl.float32)
            xhi = tl.load(q_ptr + off + 64, mask=m2, other=0.0).to(tl.float32)
            olo, ohi = _norm_rope(xlo, xhi, wlo, whi, c, s, eps)
            tl.store(qo_ptr + off, olo, mask=m2)
            tl.store(qo_ptr + off + 64, ohi, mask=m2)
    else:
        # ------------------------------------------------- key / value heads
        kc = hc - NQC
        if EXACT:
            cp = tl.load(cp_ptr + offs_s).to(tl.int64)
        else:
            cp = tl.load(cp_ptr + offs_s, mask=m, other=0).to(tl.int64)
        wlo = tl.load(kw_ptr + d).to(tl.float32)
        whi = tl.load(kw_ptr + 64 + d).to(tl.float32)
        bl = b.to(tl.int64)
        kbase = b * sk0 + kc * (HK * sk1) + offs_s[:, None] * sk2 + d[None, :]
        vbase = b * sv0 + kc * (HK * sv1) + offs_s[:, None] * sv2 + d[None, :]
        # cache offsets must be int64: b*skc0 alone reaches ~1.9e9 at B=8
        kcb = bl * skc0 + kc * (HK * skc1) + cp[:, None] * skc2 + d[None, :]
        vcb = bl * svc0 + kc * (HK * svc1) + cp[:, None] * svc2 + d[None, :]
        for i in tl.range(0, HK, num_stages=NS):
            off = kbase + i * sk1
            xlo = tl.load(k_ptr + off, mask=m2, other=0.0).to(tl.float32)
            xhi = tl.load(k_ptr + off + 64, mask=m2, other=0.0).to(tl.float32)
            olo, ohi = _norm_rope(xlo, xhi, wlo, whi, c, s, eps)
            tl.store(ko_ptr + off, olo, mask=m2)
            tl.store(ko_ptr + off + 64, ohi, mask=m2)

            coff = kcb + i * skc1
            tl.store(kc_ptr + coff, olo, mask=m2)
            tl.store(kc_ptr + coff + 64, ohi, mask=m2)

            voff = vbase + i * sv1
            vlo = tl.load(v_ptr + voff, mask=m2, other=0.0)
            vhi = tl.load(v_ptr + voff + 64, mask=m2, other=0.0)
            vcoff = vcb + i * svc1
            tl.store(vc_ptr + vcoff, vlo, mask=m2)
            tl.store(vc_ptr + vcoff + 64, vhi, mask=m2)


# ---------------------------------------------------------------------------
# configuration selection
# ---------------------------------------------------------------------------

_QDIV = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96)
_NCU = 256
_FORCE = os.environ.get("SOLX_FORCE")


def _plan(B, S):
    """Choose (BS, HQ, HK, num_warps, num_stages) for this shape."""
    if _FORCE:
        p = _FORCE.split(",")
        return int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4])

    best = None
    for BS in (1, 2, 4, 8, 16, 32, 64):
        if BS > 1 and BS > 2 * S:
            continue
        num_sb = (S + BS - 1) // BS
        for HQ in _QDIV:
            NQC = 96 // HQ
            for HK in (1, 2, 4, 8):
                NKC = 8 // HK
                progs = B * num_sb * (NQC + NKC)
                if progs < 2 * _NCU and progs < B * S * 104 // max(1, BS):
                    pass
                # want >= ~4 workgroups/CU, then the fattest tile
                fill = min(progs / (4.0 * _NCU), 1.0)
                key = (round(fill, 3), BS * HQ, BS)
                if best is None or key > best[0]:
                    best = (key, BS, HQ, HK)
    _, BS, HQ, HK = best
    elems = BS * 64
    nw = 1 if elems <= 128 else (2 if elems <= 256 else 4)
    return BS, HQ, HK, nw, 1


# ---------------------------------------------------------------------------
# launch: Triton's generic dispatch path costs ~30us of CPU time for a kernel
# with this many arguments, which dwarfs the GPU work on the small shapes.
# We therefore cache the compiled kernel per shape and re-invoke it directly,
# substituting only the tensor arguments.  Guarded by a full fallback: any
# deviation in Triton's internals and we silently use the normal path.
# ---------------------------------------------------------------------------

_cache = {}
_NTENSOR = 12


class _Plan:
    __slots__ = ("grid", "args", "consts", "fast", "crun", "func", "pmeta",
                 "stream", "getstream")

    def __init__(self, grid, args, consts):
        self.grid = grid
        self.args = args
        self.consts = consts
        self.fast = False


def _build(key, B, S, tensors, eps, strides):
    BS, HQ, HK, nw, ns = _plan(B, S)
    NQC, NKC = 96 // HQ, 8 // HK
    num_sb = (S + BS - 1) // BS
    NCH = NQC + NKC
    grid = B * num_sb * NCH
    exact = (S % BS == 0)

    args = list(tensors) + [float(eps), S, num_sb, NCH, NQC] + list(strides)
    consts = dict(HQ=HQ, HK=HK, BS=BS, EXACT=exact, NS=ns,
                  num_warps=nw, num_stages=1)

    compiled = _fused_kernel[(grid,)](*args, **consts)

    p = _Plan(grid, args, consts)
    # try to install the low-overhead launch path
    try:
        from triton.runtime import driver
        drv = driver.active
        dev = drv.get_current_device()
        binder = _fused_kernel.device_caches[dev][4]
        bound, _spec, _opt = binder(*args, **consts)
        bv = list(bound.values())
        # the first _NTENSOR bound values must be exactly our tensors, else the
        # positional substitution below would corrupt the launch
        aligned = 0
        for t in tensors:
            aligned |= t.data_ptr()
        if all(bv[i] is tensors[i] for i in range(_NTENSOR)) and (aligned & 15) == 0:
            p.args = bv
            p.crun = compiled.run
            p.func = compiled.function
            p.pmeta = compiled.packed_metadata
            p.getstream = drv.get_current_stream
            p.stream = dev
            p.fast = True
    except Exception:
        p.fast = False
    _cache[key] = p
    return p


@torch.no_grad()
def run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position_ids: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    rms_norm_eps: float,
):
    B, NQ, S, D = query.shape

    query_rotated = torch.empty_like(query)
    key_rotated = torch.empty_like(key)

    strides = (
        query.stride(0), query.stride(1), query.stride(2),
        key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        position_ids.stride(0), position_ids.stride(1),
    )
    tensors = (query, key, value, position_ids, key_cache, value_cache,
               cache_position, q_norm_weight, k_norm_weight, inv_freq,
               query_rotated, key_rotated)
    key_t = (B, S, strides, float(rms_norm_eps))

    p = _cache.get(key_t)
    if p is None:
        p = _build(key_t, B, S, tensors, rms_norm_eps, strides)
        return query_rotated, key_rotated, key_cache, value_cache

    a = p.args
    a[0] = query
    a[1] = key
    a[2] = value
    a[3] = position_ids
    a[4] = key_cache
    a[5] = value_cache
    a[6] = cache_position
    a[7] = q_norm_weight
    a[8] = k_norm_weight
    a[9] = inv_freq
    a[10] = query_rotated
    a[11] = key_rotated

    # Triton specializes the compiled code on 16B pointer alignment, so the
    # cached binary is only valid for equally-aligned buffers.  Torch's caching
    # allocator returns >=256B-aligned blocks, so this holds in practice, but
    # check rather than assume -- a mismatch would silently miscompute.
    if p.fast and ((query.data_ptr() | key.data_ptr() | value.data_ptr()
                    | key_cache.data_ptr() | value_cache.data_ptr()
                    | position_ids.data_ptr() | cache_position.data_ptr()
                    | q_norm_weight.data_ptr() | k_norm_weight.data_ptr()
                    | inv_freq.data_ptr() | query_rotated.data_ptr()
                    | key_rotated.data_ptr()) & 15) == 0:
        # fetch the live stream each call (~0.15us) so we honour whatever
        # stream the caller is on rather than baking one in
        p.crun(p.grid, 1, 1, p.getstream(p.stream), p.func, p.pmeta,
               None, None, None, *a)
    else:
        _fused_kernel[(p.grid,)](*a, **p.consts)

    return query_rotated, key_rotated, key_cache, value_cache
