import math

import torch
import triton
import triton.language as tl

NEG = tl.constexpr(-3.0e38)


@triton.jit
def _prep_kernel(qo_indptr, blk_indptr, B, BLOCK_Q: tl.constexpr, MAXB: tl.constexpr):
    offs = tl.arange(0, MAXB)
    m = offs < B
    v1 = tl.load(qo_indptr + offs + 1, mask=m, other=0)
    v0 = tl.load(qo_indptr + offs, mask=m, other=0)
    n = tl.where(m, tl.cdiv(v1 - v0, BLOCK_Q), 0)
    c = tl.cumsum(n, 0)
    tl.store(blk_indptr + offs + 1, c, mask=m)
    tl.store(blk_indptr + offs, 0, mask=offs == 0)


@triton.jit
def _mla_kernel(
    q_nope,
    q_pe,
    ckv,
    kpe,
    qo_indptr,
    kv_indptr,
    kv_indices,
    blk_indptr,
    Out,
    Lse,
    qk_scale,
    B,
    H: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAXB: tl.constexpr,
):
    pid = tl.program_id(0)
    ntot = tl.load(blk_indptr + B)
    if pid >= ntot:
        return

    offs_b = tl.arange(0, MAXB)
    bv = tl.load(blk_indptr + offs_b, mask=offs_b <= B, other=2147483647)
    b = tl.sum(tl.where(bv <= pid, 1, 0)) - 1

    blk = pid - tl.load(blk_indptr + b)
    q_start = tl.load(qo_indptr + b)
    q_end = tl.load(qo_indptr + b + 1)
    kb = tl.load(kv_indptr + b)
    ke = tl.load(kv_indptr + b + 1)
    kv_len = ke - kb
    q_len = q_end - q_start
    prefix = kv_len - q_len

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_dp = tl.arange(0, DP)

    tok_local = blk * BLOCK_Q + offs_m // H
    row = (q_start + blk * BLOCK_Q).to(tl.int64) * H + offs_m
    rmask = tok_local < q_len
    pos = prefix + tok_local

    q_n = tl.load(
        q_nope + row[:, None] * D + offs_d[None, :],
        mask=rmask[:, None],
        other=0.0,
    )
    q_p = tl.load(
        q_pe + row[:, None] * DP + offs_dp[None, :],
        mask=rmask[:, None],
        other=0.0,
    )

    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # keys with index <= min(pos) over the block are unmasked for every row
    lo_pos = prefix + blk * BLOCK_Q
    n_full = tl.minimum(kv_len, tl.maximum(lo_pos + 1, 0))
    n_full = (n_full // BLOCK_N) * BLOCK_N
    hi_pos = prefix + blk * BLOCK_Q + BLOCK_Q
    kv_hi = tl.minimum(kv_len, tl.maximum(hi_pos, 0))

    offs_n0 = tl.arange(0, BLOCK_N)

    for start_n in range(0, n_full, BLOCK_N):
        offs_n = start_n + offs_n0
        idx = tl.load(kv_indices + kb + offs_n).to(tl.int64)
        kc = tl.load(ckv + idx[:, None] * D + offs_d[None, :])
        kp = tl.load(kpe + idx[:, None] * DP + offs_dp[None, :])
        s = tl.dot(q_n, tl.trans(kc))
        s += tl.dot(q_p, tl.trans(kp))
        s = s * qk_scale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(kc.dtype), kc)
        m_i = m_new

    for start_n in range(n_full, kv_hi, BLOCK_N):
        offs_n = start_n + offs_n0
        nmask = offs_n < kv_hi
        idx = tl.load(kv_indices + kb + offs_n, mask=nmask, other=0).to(tl.int64)
        kc = tl.load(
            ckv + idx[:, None] * D + offs_d[None, :], mask=nmask[:, None], other=0.0
        )
        kp = tl.load(
            kpe + idx[:, None] * DP + offs_dp[None, :], mask=nmask[:, None], other=0.0
        )
        s = tl.dot(q_n, tl.trans(kc))
        s += tl.dot(q_p, tl.trans(kp))
        s = s * qk_scale
        s = tl.where(nmask[None, :] & (offs_n[None, :] <= pos[:, None]), s, NEG)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(kc.dtype), kc)
        m_i = m_new

    empty = l_i == 0.0
    l_safe = tl.where(empty, 1.0, l_i)
    out = acc / l_safe[:, None]
    out = tl.where(empty[:, None], 0.0, out)
    lse = tl.where(empty, -float("inf"), m_i + tl.log2(l_safe))

    tl.store(
        Out + row[:, None] * D + offs_d[None, :],
        out.to(Out.dtype.element_ty),
        mask=rmask[:, None],
    )
    tl.store(Lse + row, lse, mask=rmask)


def _next_pow2(x):
    return 1 << (x - 1).bit_length() if x > 1 else 1


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    len_indptr = qo_indptr.shape[0]
    batch_size = len_indptr - 1
    device = q_nope.device

    output = torch.empty(
        (total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

    if total_q == 0 or batch_size <= 0:
        return output, lse

    BLOCK_Q = 2
    BLOCK_M = BLOCK_Q * num_qo_heads
    BLOCK_N = 64
    MAXB = _next_pow2(batch_size + 1)

    blk_indptr = torch.empty((batch_size + 1,), dtype=torch.int32, device=device)
    _prep_kernel[(1,)](qo_indptr, blk_indptr, batch_size, BLOCK_Q=BLOCK_Q, MAXB=MAXB)

    grid = (triton.cdiv(total_q, BLOCK_Q) + batch_size,)
    _mla_kernel[grid](
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        blk_indptr,
        output,
        lse,
        float(sm_scale) * 1.4426950408889634,
        batch_size,
        H=num_qo_heads,
        D=head_dim_ckv,
        DP=head_dim_kpe,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        MAXB=MAXB,
        num_warps=4,
        num_stages=1,
    )
    return output, lse
