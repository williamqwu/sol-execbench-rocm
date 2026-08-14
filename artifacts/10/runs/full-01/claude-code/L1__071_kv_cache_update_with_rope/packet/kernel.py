import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fused "RoPE the new keys + concatenate onto the KV cache".
#
# The reference performs, per call:
#     k_rot = (k * cos) + (rotate_half(k) * sin)      # bf16 arithmetic
#     out_k = cat([key_cache,   k_rot],        dim=2)
#     out_v = cat([value_cache, value_states], dim=2)
#
# which materialises the rotated keys, then copies everything again for each
# cat.  One pass is enough: every element of the (B,H,S_tot,D) output is either
# a verbatim copy from the cache or a RoPE-transformed new key, so a single
# kernel writes both outputs with the workgroup index selecting the region.
#
# The workloads span 2.5 KB .. 168 MB.  The large ones are HBM-bound (they run
# at ~6.4 TB/s, the measured copy ceiling on this part); the small ones are
# entirely launch-bound, so the per-call CPU cost is what matters.  Triton's
# Python dispatch costs ~16-21us per launch for a kernel with 8 pointers; the
# compiled kernel's C launcher costs ~3.6us.  We bind the latter directly,
# behind a bitwise self-test and an alignment guard, with a fallback to the
# ordinary path if anything is off.
# ---------------------------------------------------------------------------


@triton.jit
def _kv_rope_cat(
    K_new,      # (B, H, S_new, D)   bf16
    V_new,      # (B, H, S_new, D)   bf16
    COS,        # (B, 1, S_new, D)   bf16
    SIN,        # (B, 1, S_new, D)   bf16
    K_cache,    # (B, H, S_cur, D)   bf16
    V_cache,    # (B, H, S_cur, D)   bf16
    OK,         # (B, H, S_tot, D)   bf16
    OV,         # (B, H, S_tot, D)   bf16
    n_cache_blk,        # workgroups (per bh) assigned to the copy region
    cache_elems,        # S_cur * D
    new_half,           # S_new * D // 2
    tot_elems,          # S_tot * D
    H: tl.constexpr,
    HALF: tl.constexpr,     # D // 2
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    blk = tl.program_id(0)
    bh = tl.program_id(1)
    out_base = bh * tot_elems

    if blk < n_cache_blk:
        # ---- existing cache: verbatim copy into the head of the output -----
        oc = blk * BLOCK_C + tl.arange(0, BLOCK_C)
        mc = oc < cache_elems
        sc = bh * cache_elems + oc
        tl.store(OK + out_base + oc, tl.load(K_cache + sc, mask=mc), mask=mc)
        tl.store(OV + out_base + oc, tl.load(V_cache + sc, mask=mc), mask=mc)
    else:
        # ---- new tokens ----------------------------------------------------
        # Indexed by half-row: `lo` walks d in [0, HALF), `hi` walks d in
        # [HALF, D).  Together they cover every element exactly once, so each
        # input byte is read once -- no second pass to fetch the rotation
        # partner, because rotate_half pairs d with d+HALF, which is exactly
        # the (lo, hi) pair a single thread already holds.
        hj = (blk - n_cache_blk) * BLOCK_N + tl.arange(0, BLOCK_N)
        mn = hj < new_half
        lo = hj + (hj // HALF) * HALF       # row*D + d
        hi = lo + HALF

        kb = bh * (new_half * 2)
        cb = (bh // H) * (new_half * 2)     # cos/sin are broadcast over heads
        db = out_base + cache_elems

        klo = tl.load(K_new + kb + lo, mask=mn, other=0.0).to(tl.float32)
        khi = tl.load(K_new + kb + hi, mask=mn, other=0.0).to(tl.float32)
        clo = tl.load(COS + cb + lo, mask=mn, other=0.0).to(tl.float32)
        chi = tl.load(COS + cb + hi, mask=mn, other=0.0).to(tl.float32)
        slo = tl.load(SIN + cb + lo, mask=mn, other=0.0).to(tl.float32)
        shi = tl.load(SIN + cb + hi, mask=mn, other=0.0).to(tl.float32)

        dt = OK.dtype.element_ty
        # Mirror the reference's rounding exactly: each bf16*bf16 product is
        # rounded back to bf16 before the add, and the sum is rounded again.
        #   out[:HALF] = bf16(k_lo*cos_lo) + bf16(-k_hi*sin_lo)
        #   out[HALF:] = bf16(k_hi*cos_hi) + bf16( k_lo*sin_hi)
        olo = (klo * clo).to(dt).to(tl.float32) + ((-khi) * slo).to(dt).to(tl.float32)
        ohi = (khi * chi).to(dt).to(tl.float32) + (klo * shi).to(dt).to(tl.float32)
        tl.store(OK + db + lo, olo.to(dt), mask=mn)
        tl.store(OK + db + hi, ohi.to(dt), mask=mn)

        # values pass through untouched, on the same index pattern
        tl.store(OV + db + lo, tl.load(V_new + kb + lo, mask=mn), mask=mn)
        tl.store(OV + db + hi, tl.load(V_new + kb + hi, mask=mn), mask=mn)


# --------------------------- launch configuration --------------------------

_TARGET_WG = 2048       # covers the 256 CUs several times over


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def _pick(elems_per_bh, bh, lo, hi):
    """Block size for one region: aim at ~_TARGET_WG workgroups overall while
    keeping enough elements per thread to saturate HBM."""
    if elems_per_bh <= 0:
        return lo
    per_bh = max(1, _TARGET_WG // max(1, bh))
    b = _next_pow2(max(1, -(-elems_per_bh // per_bh)))
    return min(max(b, lo), hi)


# ----------------------- fast path: direct C launcher ----------------------
#
# JITFunction.run rebinds and re-specialises every argument on each call.  The
# CompiledKernel's launcher is a C extension taking positional args.  We build
# the argument tuple once per shape and call it directly.  The private API is
# probed once (_self_test) and every failure mode falls back to plain Triton.

_RAW_OK = None          # None = untested, True/False = self-test verdict
_DEV = None             # triton device index
_GETSTREAM = None       # driver.active.get_current_stream
_DUMMY = None           # 16B-aligned placeholder tensor for warmup
_CFG = {}


def _bootstrap():
    global _RAW_OK, _DEV, _GETSTREAM, _DUMMY
    _RAW_OK = False
    try:
        from triton.runtime import driver
        d = driver.active
        _DEV = d.get_current_device()
        _GETSTREAM = d.get_current_stream
        dev = torch.device("cuda", torch.cuda.current_device())
        _DUMMY = torch.empty(8192, dtype=torch.bfloat16, device=dev)
        _RAW_OK = _verify_raw(dev)
    except Exception:
        _RAW_OK = False
    finally:
        _CFG.clear()


def _plan(B, H, S_new, S_cur, D):
    bh = B * H
    cache_elems = S_cur * D
    new_half = S_new * D // 2
    tot_elems = (S_cur + S_new) * D
    # Swept over all 16 workload shapes: a 2048-workgroup target with the copy
    # block capped at 8192 and the RoPE block at 512 was the best static rule
    # found (117.3us summed GPU time vs 111.9us for per-shape optima).
    bc = _pick(cache_elems, bh, 256, 8192)
    bn = _pick(new_half, bh, 64, 512)
    n_cache_blk = -(-cache_elems // bc) if cache_elems else 0
    n_new_blk = -(-new_half // bn)
    nw = max(1, min(8, max(bc // 512, (bn * 2) // 512)))
    return dict(
        grid=(n_cache_blk + n_new_blk, bh, 1),
        ints=(n_cache_blk, cache_elems, new_half, tot_elems),
        H=H, HALF=D // 2, BLOCK_C=bc, BLOCK_N=bn, num_warps=nw,
    )


def _compile_raw(p):
    """Return (launch, coop, function, packed_metadata) or None."""
    try:
        ck = _kv_rope_cat.warmup(
            _DUMMY, _DUMMY, _DUMMY, _DUMMY, _DUMMY, _DUMMY, _DUMMY, _DUMMY,
            *p["ints"],
            H=p["H"], HALF=p["HALF"],
            BLOCK_C=p["BLOCK_C"], BLOCK_N=p["BLOCK_N"],
            num_warps=p["num_warps"], num_stages=1,
            grid=p["grid"],
        )
        ck._init_handles()
        L = ck.run
        if getattr(L, "profile_scratch_size", 0):
            return None            # kernel wants a scratch buffer; not our path
        return (L.launch, L.launch_cooperative_grid, ck.function,
                ck.packed_metadata)
    except Exception:
        return None


def _make_entry(B, H, S_new, S_cur, D):
    p = _plan(B, H, S_new, S_cur, D)
    raw = _compile_raw(p) if _RAW_OK else None
    g = p["grid"]
    shape = (B, H, S_cur + S_new, D)
    # One allocation carved into two views beats two allocations once the
    # buffers are large enough that the caching allocator has to do real work
    # (measured: 28.5us -> 15.4us at 42 MB/output, but slightly worse below
    # ~1 MB where the fixed cost of the extra view dominates).
    fused_alloc = (B * H * (S_cur + S_new) * D * 2) >= (1 << 20)
    e = (
        raw,                                   # 0
        g[0], g[1], g[2],                      # 1,2,3
        p["ints"],                             # 4
        (p["H"], p["HALF"], p["BLOCK_C"], p["BLOCK_N"]),   # 5
        p["num_warps"],                        # 6
        shape,                                 # 7  output shape
        (2,) + shape if fused_alloc else None,  # 8  fused alloc shape
    )
    _CFG[(B, H, S_new, S_cur, D)] = e
    return e


def run(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
):
    if _RAW_OK is None:
        _bootstrap()

    B, H, S_new, D = key_states.shape
    S_cur = key_cache.shape[2]

    key = (B, H, S_new, S_cur, D)
    e = _CFG.get(key)
    if e is None:
        e = _make_entry(B, H, S_new, S_cur, D)

    # the kernel walks flat memory, so it needs dense row-major inputs
    if not (key_states.is_contiguous() and value_states.is_contiguous()
            and cos.is_contiguous() and sin.is_contiguous()
            and key_cache.is_contiguous() and value_cache.is_contiguous()):
        # only here can an autograd-tracked op appear, so scope the guard to
        # it rather than paying ~1.1us for a no_grad wrapper on every call
        with torch.no_grad():
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
            cos = cos.contiguous()
            sin = sin.contiguous()
            key_cache = key_cache.contiguous()
            value_cache = value_cache.contiguous()

    fs = e[8]
    if fs is None:
        ok = torch.empty(e[7], dtype=key_states.dtype, device=key_states.device)
        ov = torch.empty_like(ok)
    else:
        both = torch.empty(fs, dtype=key_states.dtype,
                           device=key_states.device)
        ok = both[0]
        ov = both[1]

    raw = e[0]
    if raw is not None:
        p0 = key_states.data_ptr(); p1 = value_states.data_ptr()
        p2 = cos.data_ptr();        p3 = sin.data_ptr()
        p4 = key_cache.data_ptr();  p5 = value_cache.data_ptr()
        p6 = ok.data_ptr();         p7 = ov.data_ptr()
        # The compiled kernel was specialised assuming 16-byte-aligned
        # pointers.  Honour that assumption or take the general path.
        if not ((p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7) & 15):
            i = e[4]; c = e[5]
            try:
                raw[0](raw[1], e[1], e[2], e[3], _GETSTREAM(_DEV), raw[2],
                       None, raw[3], None, None, None,
                       p0, p1, p2, p3, p4, p5, p6, p7,
                       i[0], i[1], i[2], i[3], c[0], c[1], c[2], c[3])
                return ok, ov
            except Exception:
                _CFG[key] = e = (None,) + e[1:]

    i = e[4]; c = e[5]
    _kv_rope_cat[(e[1], e[2], e[3])](
        key_states, value_states, cos, sin, key_cache, value_cache, ok, ov,
        i[0], i[1], i[2], i[3],
        H=c[0], HALF=c[1], BLOCK_C=c[2], BLOCK_N=c[3],
        num_warps=e[6], num_stages=1,
    )
    return ok, ov


# ------------------------------- self test ---------------------------------


def _verify_raw(dev):
    """Run one shape through the direct launcher and demand bitwise agreement
    with the reference semantics.  Anything else disables the fast path."""
    B, H, S, C, D = 2, 10, 3, 5, 128
    g = torch.Generator(device=dev).manual_seed(1234)
    f = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=torch.bfloat16)
    ks, vs = f(B, H, S, D), f(B, H, S, D)
    cs, sn = f(B, 1, S, D), f(B, 1, S, D)
    kc, vc = f(B, H, C, D), f(B, H, C, D)

    half = D // 2
    k_rot = torch.cat((-ks[..., half:], ks[..., :half]), dim=-1)
    want_k = torch.cat([kc, (ks * cs) + (k_rot * sn)], dim=2)
    want_v = torch.cat([vc, vs], dim=2)

    p = _plan(B, H, S, C, D)
    raw = _compile_raw(p)
    if raw is None:
        return False
    gk, gv = torch.empty_like(want_k), torch.empty_like(want_v)
    gr, i, cst = p["grid"], p["ints"], (p["H"], p["HALF"], p["BLOCK_C"],
                                        p["BLOCK_N"])
    raw[0](raw[1], gr[0], gr[1], gr[2], _GETSTREAM(_DEV), raw[2],
           None, raw[3], None, None, None,
           ks.data_ptr(), vs.data_ptr(), cs.data_ptr(), sn.data_ptr(),
           kc.data_ptr(), vc.data_ptr(), gk.data_ptr(), gv.data_ptr(),
           i[0], i[1], i[2], i[3], cst[0], cst[1], cst[2], cst[3])
    torch.cuda.synchronize()
    return bool(torch.equal(gk, want_k) and torch.equal(gv, want_v))
