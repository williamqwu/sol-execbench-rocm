import os

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_kernel(
    Q, KC, VC, QO_INDPTR, KV_INDPTR, KV_INDICES,
    Out, LSE,
    qk_scale,
    NB,
    stride_qt, stride_qh,
    stride_kp, stride_kh,
    stride_vp, stride_vh,
    stride_ot, stride_oh,
    stride_lt,
    GQA: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
    BNB: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    # --- inline tile map: prefix-sum of per-sequence tile counts ---
    nb_off = tl.arange(0, BNB)
    nb_m = nb_off < NB
    qs_v = tl.load(QO_INDPTR + nb_off, mask=nb_m, other=0)
    qe_v = tl.load(QO_INDPTR + nb_off + 1, mask=nb_m, other=0)
    tiles_v = (qe_v - qs_v + (BM - 1)) // BM
    tiles_v = tl.where(nb_m, tiles_v, 0)
    cum_v = tl.cumsum(tiles_v, 0)

    total_tiles = tl.sum(tiles_v, 0)
    if t >= total_tiles:
        return

    # b = number of sequences whose cumulative tile count is <= t
    b = tl.sum(((cum_v <= t) & nb_m).to(tl.int32), 0)
    prev = tl.sum(tl.where(nb_off < b, tiles_v, 0), 0)
    tile_local = t - prev

    q_start = tl.sum(tl.where(nb_off == b, qs_v, 0), 0)
    q_end = tl.sum(tl.where(nb_off == b, qe_v, 0), 0)
    kv_start = tl.load(KV_INDPTR + b)
    kv_end = tl.load(KV_INDPTR + b + 1)

    qlen = q_end - q_start
    kvlen = kv_end - kv_start

    offs_m = tile_local * BM + tl.arange(0, BM)   # local query index
    rows_valid = offs_m < qlen
    offs_d = tl.arange(0, D)

    kv_head = h // GQA

    delta = kvlen - qlen
    last_q = tl.minimum(tile_local * BM + BM - 1, qlen - 1)
    hi_kv = tl.minimum(last_q + 1 + delta, kvlen)

    # If every row of this tile is fully causal-masked there is nothing to read:
    # predicate the Q load off so the tile costs only its zero/-inf stores.
    q_ptrs = Q + (q_start + offs_m)[:, None] * stride_qt + h * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=rows_valid[:, None] & (hi_kv > 0), other=0.0)

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)

    # per-row exclusive upper bound on kv index (causal)
    row_hi = tl.minimum(offs_m + 1 + delta, kvlen)

    if hi_kv > 0:
        for start_n in range(0, hi_kv, BN):
            offs_n = start_n + tl.arange(0, BN)
            n_valid = offs_n < hi_kv
            pid = tl.load(KV_INDICES + kv_start + offs_n, mask=n_valid, other=0)
            k_ptrs = KC + pid[:, None] * stride_kp + kv_head * stride_kh + offs_d[None, :]
            k = tl.load(k_ptrs, mask=n_valid[:, None], other=0.0)
            v_ptrs = VC + pid[:, None] * stride_vp + kv_head * stride_vh + offs_d[None, :]
            v = tl.load(v_ptrs, mask=n_valid[:, None], other=0.0)

            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale
            keep = (offs_n[None, :] < row_hi[:, None]) & rows_valid[:, None]
            qk = tl.where(keep, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp2(m_i - m_safe)
            p = tl.exp2(qk - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            # split p into two bf16 terms so the PV product keeps ~fp32 accuracy
            p_hi = p.to(v.dtype)
            p_lo = (p - p_hi.to(tl.float32)).to(v.dtype)
            acc = acc * alpha[:, None] + tl.dot(p_hi, v, out_dtype=tl.float32) \
                + tl.dot(p_lo, v, out_dtype=tl.float32)
            m_i = m_new

    empty = l_i == 0.0
    l_safe = tl.where(empty, 1.0, l_i)
    out = acc / l_safe[:, None]
    out = tl.where(empty[:, None], 0.0, out)
    lse = tl.where(empty, float("-inf"), m_i + tl.log2(l_safe))

    o_ptrs = Out + (q_start + offs_m)[:, None] * stride_ot + h * stride_oh + offs_d[None, :]
    tl.store(o_ptrs, out.to(Out.dtype.element_ty), mask=rows_valid[:, None])
    l_ptrs = LSE + (q_start + offs_m) * stride_lt + h
    tl.store(l_ptrs, lse, mask=rows_valid)


LOG2E = 1.4426950408889634

BM = int(os.environ.get("TBM", 64))
BN = int(os.environ.get("TBN", 64))
NW = int(os.environ.get("TNW", 2))
NS = int(os.environ.get("TNS", 1))


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    nb = qo_indptr.shape[0] - 1

    output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)
    if total_q == 0 or nb <= 0:
        return output, lse

    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.item())

    max_tiles = (total_q + BM - 1) // BM + nb
    bnb = max(16, triton.next_power_of_2(nb))

    kc = k_cache.view(num_pages, num_kv_heads, head_dim)
    vc = v_cache.view(num_pages, num_kv_heads, head_dim)

    _attn_kernel[(max_tiles, num_qo_heads)](
        q, kc, vc, qo_indptr, kv_indptr, kv_indices,
        output, lse,
        sm_scale * LOG2E,
        nb,
        q.stride(0), q.stride(1),
        kc.stride(0), kc.stride(1),
        vc.stride(0), vc.stride(1),
        output.stride(0), output.stride(1),
        lse.stride(0),
        GQA=num_qo_heads // num_kv_heads,
        BM=BM, BN=BN, D=head_dim, BNB=bnb,
        num_warps=NW, num_stages=NS,
    )
    return output, lse
