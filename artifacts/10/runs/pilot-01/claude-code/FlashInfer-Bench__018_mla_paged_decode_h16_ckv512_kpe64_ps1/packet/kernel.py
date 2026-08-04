"""MLA paged decode (h16, ckv512, kpe64, page_size=1) for MI355X / gfx950.

Structure
---------
Flash-decoding: one split-K pass over the KV pages producing per-split
(output, running LSE), then a combine pass that merges the splits.

Three things carry the performance here, in order of impact:

1. **Transposed accumulator.**  The K tile is loaded as ``[D, N]`` and the
   output accumulator is kept as ``[D, H]``, so the PV dot reuses the K tile
   in the exact register layout it arrived in.  The alternative transposes a
   ``[N, 512]`` bf16 tile every iteration; here only the small ``[H, N]``
   probability tile is transposed.

2. **Launch-cost elimination.**  At these sizes the GPU work is 10-40 us and
   Triton's Python binder costs ~9 us *per launch*.  We resolve the compiled
   kernel once per (config, dtype) and thereafter call its C launcher
   directly, which measured 4.9 us vs 18.0 us for the same launch.  This is
   Triton's own launch path with the redundant re-binding hoisted out; the
   `_slow_launch` fallback runs the ordinary `jf[grid](...)` path if anything
   about the fast path does not resolve.

3. **Workspace reuse.**  The fp32 split buffers are pooled across calls so a
   steady-state call does no allocation.  Only scratch is pooled -- outputs
   are freshly allocated every call.
"""

import torch
import triton
import triton.language as tl

from triton.runtime import driver

_LOG2E = 1.4426950408889634


@triton.jit
def _mla_decode_split(
    q_nope_ptr,
    q_pe_ptr,
    ckv_ptr,
    kpe_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    partial_o_ptr,   # [B, S, H, D] fp32, or [B, H, D] bf16 when WRITE_FINAL
    partial_e_ptr,   # [B, S, H]   fp32, or [B, H]    fp32 when WRITE_FINAL
    scale_log2e,     # sm_scale * log2(e)
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WRITE_FINAL: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)

    beg = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    L = end - beg

    n_blocks = tl.cdiv(L, BLOCK_N)
    per = tl.cdiv(n_blocks, S)
    blk_beg = s * per
    blk_end = tl.minimum(blk_beg + per, n_blocks)

    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_dp = tl.arange(0, DP)

    q_n = tl.load(q_nope_ptr + b * (H * D) + offs_h[:, None] * D + offs_d[None, :])
    q_p = tl.load(q_pe_ptr + b * (H * DP) + offs_h[:, None] * DP + offs_dp[None, :])

    m_i = tl.full([H], -1.0e38, dtype=tl.float32)
    l_i = tl.zeros([H], dtype=tl.float32)
    # Accumulator transposed [D, H]: lets the PV dot consume the K tile in the
    # same [D, N] layout it was loaded in, so no in-register transpose of the
    # large bf16 K tile is needed -- only of the small [H, N] probabilities.
    accT = tl.zeros([D, H], dtype=tl.float32)

    idx_base = kv_indices_ptr + beg

    for blk in range(blk_beg, blk_end):
        offs_n = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < L
        idx = tl.load(idx_base + offs_n, mask=n_mask, other=0).to(tl.int64)

        kcT = tl.load(
            ckv_ptr + idx[None, :] * D + offs_d[:, None],
            mask=n_mask[None, :],
            other=0.0,
        )
        kpT = tl.load(
            kpe_ptr + idx[None, :] * DP + offs_dp[:, None],
            mask=n_mask[None, :],
            other=0.0,
        )

        qk = tl.dot(q_n, kcT) + tl.dot(q_p, kpT)
        qk = qk * scale_log2e
        qk = tl.where(n_mask[None, :], qk, -1.0e38)

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        accT = accT * alpha[None, :] + tl.dot(kcT, tl.trans(p).to(kcT.dtype))
        m_i = m_new

    empty = l_i == 0.0
    l_safe = tl.where(empty, 1.0, l_i)
    accT = accT / l_safe[None, :]
    e_i = tl.where(empty, float("-inf"), m_i + tl.log2(l_safe))
    acc = tl.trans(accT)

    if WRITE_FINAL:
        tl.store(
            partial_o_ptr + b * (H * D) + offs_h[:, None] * D + offs_d[None, :],
            acc.to(partial_o_ptr.dtype.element_ty),
        )
        tl.store(partial_e_ptr + b * H + offs_h, e_i)
    else:
        tl.store(
            partial_o_ptr + (b * S + s) * (H * D) + offs_h[:, None] * D + offs_d[None, :],
            acc,
        )
        tl.store(partial_e_ptr + (b * S + s) * H + offs_h, e_i)


@triton.jit
def _mla_combine(
    partial_o_ptr,   # [B, S, H, D] fp32
    partial_e_ptr,   # [B, S, H] fp32
    out_ptr,         # [B, H, D] bf16
    lse_ptr,         # [B, H] fp32
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)

    offs_s = tl.arange(0, BLOCK_S)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)

    s_mask = offs_s < S
    e = tl.load(                                        # [S, H]
        partial_e_ptr + (b * S + offs_s[:, None]) * H + offs_h[None, :],
        mask=s_mask[:, None],
        other=float("-inf"),
    )

    m = tl.max(e, 0)                                    # [H]
    is_empty = m == float("-inf")
    m_safe = tl.where(is_empty, 0.0, m)
    w = tl.where(e == float("-inf"), 0.0, tl.exp2(e - m_safe[None, :]))
    w = tl.where(s_mask[:, None], w, 0.0)
    tot = tl.sum(w, 0)                                  # [H]
    tot_safe = tl.where(tot == 0.0, 1.0, tot)

    acc = tl.zeros([H, D], dtype=tl.float32)
    for j in range(0, BLOCK_S):
        if j < S:
            po = tl.load(
                partial_o_ptr
                + (b * S + j) * (H * D)
                + offs_h[:, None] * D
                + offs_d[None, :]
            )
            wj = tl.sum(tl.where(offs_s[:, None] == j, w, 0.0), 0)   # [H]
            acc += po * wj[:, None]

    acc = acc / tot_safe[:, None]
    tl.store(
        out_ptr + b * (H * D) + offs_h[:, None] * D + offs_d[None, :],
        acc.to(out_ptr.dtype.element_ty),
    )
    tl.store(
        lse_ptr + b * H + offs_h,
        tl.where(is_empty, float("-inf"), m_safe + tl.log2(tot_safe)),
    )


# --------------------------------------------------------------------------
# Fast launch path
# --------------------------------------------------------------------------
# Triton's `jf[grid](*args, **kw)` re-runs the argument binder and cache-key
# computation on every call (~9 us here, against 10-40 us of GPU work). We
# resolve the CompiledKernel once and afterwards invoke its C launcher
# directly, in exactly the way `JITFunction.run` would.

_LAUNCHERS = {}


def _resolve(jf, grid, kwargs, args):
    """Compile/lookup once, return (run, function, packed_metadata) or None."""
    try:
        dev = driver.active.get_current_device()
        kcache = jf.device_caches[dev][0]
        before = set(kcache.keys())
        jf[grid](*args, **kwargs)
        fresh = [k for k in kcache.keys() if k not in before]
        ck = kcache[fresh[0]] if fresh else None
        if ck is None:
            # already compiled on a previous call under a key we can't
            # cheaply recompute -- find it by matching metadata
            cands = list(kcache.values())
            if len(cands) != 1:
                return None
            ck = cands[0]
        ck._init_handles()
        if getattr(ck, "metadata", None) is not None:
            if getattr(ck.metadata, "profile_scratch_size", 0):
                return None            # needs scratch allocation; use slow path
        return (ck.run, ck.function, ck.packed_metadata)
    except Exception:
        return None


_STREAM = None


def _stream():
    global _STREAM
    if _STREAM is None:
        _STREAM = driver.active.get_current_stream(driver.active.get_current_device())
    return _STREAM


def _launch(jf, key, grid, kwargs, run_args, const_args):
    """Launch `jf`, using the cached C launcher when available."""
    ent = _LAUNCHERS.get(key)
    if ent is None:
        full = dict(kwargs)
        ent = _resolve(jf, grid, full, run_args)
        _LAUNCHERS[key] = ent if ent is not None else False
        if ent is not None:
            return                      # _resolve already ran it once
    if ent is False or ent is None:
        jf[grid](*run_args, **kwargs)
        return
    run, fn, pm = ent
    try:
        run(grid[0], grid[1], 1, _stream(), fn, pm, None, None, None,
            *run_args, *const_args)
    except TypeError:
        _LAUNCHERS[key] = False
        jf[grid](*run_args, **kwargs)


# --------------------------------------------------------------------------
# Scratch pool -- only intermediates, never outputs.
# --------------------------------------------------------------------------
_WS = {}


def _scratch(key, shape, dtype, device):
    numel = 1
    for s in shape:
        numel *= s
    buf = _WS.get(key)
    if buf is None or buf.numel() < numel or buf.device != device or buf.dtype != dtype:
        buf = torch.empty(max(numel, 1), dtype=dtype, device=device)
        _WS[key] = buf
    return buf[:numel].view(*shape)


def _plan(batch_size, total_tokens):
    """(BLOCK_N, num_splits, num_warps) from the average sequence length.

    Tuned by sweep on MI355X. Two regimes:

    * Short sequences -- one program per batch element, ``S == 1``. This skips
      the combine kernel altogether, and at these sizes the ~10 us saved on
      that second launch outweighs the lost parallelism even at batch 1.
    * Long sequences -- split-K, with just enough splits to fill the 256 CUs.
      More splits than that only inflate the fp32 partial traffic the combine
      pass must read back.
    """
    avg = max(1, total_tokens // max(1, batch_size))

    if avg <= 340:
        # single pass, no combine launch
        if avg <= 32:
            return 16, 1, 8
        return 64, 1, 4

    if avg <= 1000:
        block_n, warps = 128, 8
    else:
        block_n, warps = 64, 4

    n_blocks = -(-avg // block_n)
    splits = max(4, min(12, 256 // max(1, batch_size)))
    splits = max(1, min(splits, n_blocks))
    return block_n, splits, warps


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, H, D = q_nope.shape
    DP = q_pe.shape[-1]
    device = q_nope.device

    if isinstance(sm_scale, torch.Tensor):
        sm_scale = sm_scale.item()
    scale_log2e = float(sm_scale) * _LOG2E

    output = torch.empty((batch_size, H, D), dtype=torch.bfloat16, device=device)
    lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)
    if batch_size == 0:
        return {"output": output, "lse": lse}

    ckv = ckv_cache.view(-1, D)
    kpe = kpe_cache.view(-1, DP)

    BLOCK_N, S, warps = _plan(batch_size, kv_indices.shape[0])

    if S == 1:
        kw = dict(S=1, H=H, D=D, DP=DP, BLOCK_N=BLOCK_N, WRITE_FINAL=True,
                  num_warps=warps, num_stages=1)
        _launch(
            _mla_decode_split,
            ("split", 1, H, D, DP, BLOCK_N, True, warps),
            (batch_size, 1),
            kw,
            (q_nope, q_pe, ckv, kpe, kv_indptr, kv_indices, output, lse, scale_log2e),
            (1, H, D, DP, BLOCK_N, True),
        )
        return {"output": output, "lse": lse}

    partial_o = _scratch("po", (batch_size, S, H, D), torch.float32, device)
    partial_e = _scratch("pe", (batch_size, S, H), torch.float32, device)

    kw = dict(S=S, H=H, D=D, DP=DP, BLOCK_N=BLOCK_N, WRITE_FINAL=False,
              num_warps=warps, num_stages=1)
    _launch(
        _mla_decode_split,
        ("split", S, H, D, DP, BLOCK_N, False, warps),
        (batch_size, S),
        kw,
        (q_nope, q_pe, ckv, kpe, kv_indptr, kv_indices,
         partial_o, partial_e, scale_log2e),
        (S, H, D, DP, BLOCK_N, False),
    )

    BS = triton.next_power_of_2(S)
    kw2 = dict(S=S, H=H, D=D, BLOCK_S=BS, num_warps=4, num_stages=1)
    _launch(
        _mla_combine,
        ("comb", S, H, D, BS),
        (batch_size, 1),
        kw2,
        (partial_o, partial_e, output, lse),
        (S, H, D, BS),
    )

    return {"output": output, "lse": lse}
