import torch
import triton
import triton.language as tl


@triton.jit
def _small_kv_kernel(
    q,
    k_cache,
    v_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    out,
    lse,
    sm_scale: tl.constexpr,
    len_indptr: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_b = tl.arange(0, BLOCK_B)
    q_starts = tl.load(qo_indptr + offs_b, mask=offs_b < len_indptr, other=0)
    q_ends = tl.load(qo_indptr + offs_b + 1, mask=offs_b < (len_indptr - 1), other=0)
    in_batch = (pid_q >= q_starts) & (pid_q < q_ends)
    batch = tl.sum(tl.where(in_batch, offs_b, 0), axis=0)

    q_start = tl.load(qo_indptr + batch)
    q_end = tl.load(qo_indptr + batch + 1)
    kv_start = tl.load(kv_indptr + batch)
    kv_end = tl.load(kv_indptr + batch + 1)

    q_len = q_end - q_start
    kv_len = kv_end - kv_start
    q_local = pid_q - q_start
    max_kv = q_local + 1 + kv_len - q_len
    max_kv = tl.minimum(max_kv, kv_len)

    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    kv_h = pid_h // 8

    q_vec = tl.load(q + (pid_q * 32 + pid_h) * 128 + offs_d).to(tl.float32)
    page_ids = tl.load(kv_indices + kv_start + offs_n, mask=offs_n < max_kv, other=0)
    k_vals = tl.load(
        k_cache + page_ids[:, None] * 512 + kv_h * 128 + offs_d[None, :],
        mask=offs_n[:, None] < max_kv,
        other=0.0,
    ).to(tl.float32)
    logits = tl.sum(k_vals * q_vec[None, :], axis=1) * sm_scale
    valid_n = offs_n < max_kv
    logits = tl.where(valid_n, logits, -3.4028234663852886e38)

    m = tl.max(logits, axis=0)
    p = tl.exp2((logits - m) * 1.4426950408889634)
    p = tl.where(valid_n, p, 0.0)
    denom = tl.sum(p, axis=0)
    denom_safe = tl.where(denom > 0.0, denom, 1.0)

    v_vals = tl.load(
        v_cache + page_ids[:, None] * 512 + kv_h * 128 + offs_d[None, :],
        mask=offs_n[:, None] < max_kv,
        other=0.0,
    ).to(tl.float32)
    acc = tl.sum(v_vals * p[:, None], axis=0) / denom_safe

    tl.store(out + (pid_q * 32 + pid_h) * 128 + offs_d, acc)
    lse_val = m * 1.4426950408889634 + tl.log2(denom)
    lse_val = tl.where(denom > 0.0, lse_val, -float("inf"))
    tl.store(lse + pid_q * 32 + pid_h, lse_val)


@triton.jit
def _flash_kernel(
    q,
    k_cache,
    v_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    out,
    lse,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_start = tl.load(qo_indptr + pid_b)
    q_end = tl.load(qo_indptr + pid_b + 1)
    kv_start = tl.load(kv_indptr + pid_b)
    kv_end = tl.load(kv_indptr + pid_b + 1)

    q_len = q_end - q_start
    kv_len = kv_end - kv_start
    delta = kv_len - q_len
    kv_h = pid_h // 8

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < q_len
    q_tile = tl.load(
        q + ((q_start + offs_m[:, None]) * 32 + pid_h) * 128 + offs_d[None, :],
        mask=q_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -3.4028234663852886e38, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    max_kv_for_q = tl.minimum(offs_m + 1 + delta, kv_len)

    has_q_block = pid_m * BLOCK_M < q_len
    start_n = 0
    while (start_n < kv_len) & has_q_block:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        active_n = offs_n < kv_len
        page_ids = tl.load(kv_indices + kv_start + offs_n, mask=active_n, other=0)
        k_tile = tl.load(
            k_cache + page_ids[:, None] * 512 + kv_h * 128 + offs_d[None, :],
            mask=active_n[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee") * sm_scale
        attn_mask = q_mask[:, None] & (offs_n[None, :] < max_kv_for_q[:, None])
        scores = tl.where(attn_mask, scores, -3.4028234663852886e38)

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp2((scores - m_ij[:, None]) * 1.4426950408889634)
        p = tl.where(attn_mask, p, 0.0)
        alpha = tl.exp2((m_i - m_ij) * 1.4426950408889634)
        l_ij = tl.sum(p, axis=1)

        v_tile = tl.load(
            v_cache + page_ids[:, None] * 512 + kv_h * 128 + offs_d[None, :],
            mask=active_n[:, None],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p, v_tile, input_precision="ieee")
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        start_n += BLOCK_N

    denom_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / denom_safe[:, None]
    lse_vals = m_i * 1.4426950408889634 + tl.log2(l_i)
    lse_vals = tl.where(l_i > 0.0, lse_vals, -float("inf"))

    store_mask = q_mask[:, None]
    tl.store(
        out + ((q_start + offs_m[:, None]) * 32 + pid_h) * 128 + offs_d[None, :],
        acc,
        mask=store_mask,
    )
    tl.store(lse + (q_start + offs_m) * 32 + pid_h, lse_vals, mask=q_mask)


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q = q.shape[0]
    out = torch.empty_like(q)
    lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)
    scale = float(sm_scale)

    if kv_indices.shape[0] <= 256:
        _small_kv_kernel[(total_q, 32)](
            q,
            k_cache,
            v_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            out,
            lse,
            scale,
            qo_indptr.shape[0],
            BLOCK_B=64,
            BLOCK_N=256,
            BLOCK_D=128,
            num_warps=8,
        )
    else:
        q_lens = qo_indptr[1:] - qo_indptr[:-1]
        max_q_len = int(torch.max(q_lens).item())
        batch_size = qo_indptr.shape[0] - 1
        block_m = 16
        _flash_kernel[(triton.cdiv(max_q_len, block_m), 32, batch_size)](
            q,
            k_cache,
            v_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            out,
            lse,
            scale,
            BLOCK_M=block_m,
            BLOCK_N=64,
            BLOCK_D=128,
            num_warps=8,
        )

    return out, lse
