import math

import aiter
import torch
import triton
import triton.language as tl


_LOG2E = 1.0 / math.log(2.0)


@triton.jit
def _tiny_gqa_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    SM_SCALE: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_pos = tl.program_id(0)
    kv_head = tl.program_id(1)
    h = tl.arange(0, 16)
    d = tl.arange(0, 128)
    n = tl.arange(0, BLOCK_N)

    q_heads = kv_head * 8 + h
    q = tl.load(
        q_ptr + q_pos * 32 * 128 + q_heads[:, None] * 128 + d[None, :],
        mask=h[:, None] < 8,
        other=0.0,
    )
    k = tl.load(
        k_ptr + n[:, None] * 4 * 128 + kv_head * 128 + d[None, :],
        mask=n[:, None] < NK,
        other=0.0,
    )
    scores = tl.dot(q, tl.trans(k)) * SM_SCALE
    valid_n = (n < NK) & (n < q_pos + 1 + NK - NQ)
    valid = (h[:, None] < 8) & valid_n[None, :]
    scores = tl.where(valid, scores, -float("inf"))
    row_max = tl.max(scores, axis=1)
    row_max = tl.where(h < 8, row_max, 0.0)
    p = tl.exp(scores - row_max[:, None])
    denom = tl.sum(p, axis=1)

    vv = tl.load(
        v_ptr + n[:, None] * 4 * 128 + kv_head * 128 + d[None, :],
        mask=n[:, None] < NK,
        other=0.0,
    )
    acc = tl.dot(p.to(tl.bfloat16), vv) / denom[:, None]
    tl.store(
        o_ptr + q_pos * 32 * 128 + q_heads[:, None] * 128 + d[None, :],
        acc,
        mask=h[:, None] < 8,
    )
    lse = (row_max + tl.log(denom)) * 1.4426950408889634
    tl.store(lse_ptr + q_pos * 32 + q_heads, lse, mask=h < 8)


@triton.jit
def _small_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    SM_SCALE: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // 8
    m = block * BLOCK_M + tl.arange(0, BLOCK_M)
    n = tl.arange(0, BLOCK_N)
    d = tl.arange(0, 128)

    q = tl.load(
        q_ptr + m[:, None] * 32 * 128 + q_head * 128 + d[None, :],
        mask=m[:, None] < NQ,
        other=0.0,
    )
    k = tl.load(
        k_ptr + n[:, None] * 4 * 128 + kv_head * 128 + d[None, :],
        mask=n[:, None] < NK,
        other=0.0,
    )
    scores = tl.dot(q, tl.trans(k)) * SM_SCALE
    valid = (
        (m[:, None] < NQ)
        & (n[None, :] < NK)
        & (n[None, :] < m[:, None] + 1 + NK - NQ)
    )
    scores = tl.where(valid, scores, -float("inf"))
    row_max = tl.max(scores, axis=1)
    row_max = tl.where(m < NQ, row_max, 0.0)
    p = tl.exp(scores - row_max[:, None])
    denom = tl.sum(p, axis=1)

    vv = tl.load(
        v_ptr + n[:, None] * 4 * 128 + kv_head * 128 + d[None, :],
        mask=n[:, None] < NK,
        other=0.0,
    )
    acc = tl.dot(p.to(tl.bfloat16), vv) / denom[:, None]
    tl.store(
        o_ptr + m[:, None] * 32 * 128 + q_head * 128 + d[None, :],
        acc,
        mask=m[:, None] < NQ,
    )
    lse = (row_max + tl.log(denom)) * 1.4426950408889634
    tl.store(lse_ptr + m * 32 + q_head, lse, mask=m < NQ)


@triton.jit
def _small_ragged_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    qo_indptr,
    kv_indptr,
    o_ptr,
    lse_ptr,
    SM_SCALE: tl.constexpr,
    BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    logical_block = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // 8

    running_blocks = 0
    selected = False
    q_start = 0
    q_end = 0
    k_start = 0
    k_end = 0
    local_block = 0
    for b in range(BATCH):
        this_q_start = tl.load(qo_indptr + b)
        this_q_end = tl.load(qo_indptr + b + 1)
        this_k_start = tl.load(kv_indptr + b)
        this_k_end = tl.load(kv_indptr + b + 1)
        block_count = (this_q_end - this_q_start + BLOCK_M - 1) // BLOCK_M
        take = (logical_block >= running_blocks) & (
            logical_block < running_blocks + block_count
        )
        q_start = tl.where(take, this_q_start, q_start)
        q_end = tl.where(take, this_q_end, q_end)
        k_start = tl.where(take, this_k_start, k_start)
        k_end = tl.where(take, this_k_end, k_end)
        local_block = tl.where(take, logical_block - running_blocks, local_block)
        selected = selected | take
        running_blocks += block_count

    m = q_start + local_block * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, 128)
    valid_m = selected & (m < q_end)
    q = tl.load(
        q_ptr + m[:, None] * 32 * 128 + q_head * 128 + d[None, :],
        mask=valid_m[:, None],
        other=0.0,
    )

    acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    running_max = tl.where(valid_m, -float("inf"), 0.0)
    q_len = q_end - q_start
    k_len = k_end - k_start
    start_n = 0
    while start_n < k_len:
        n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(
            k_ptr
            + (k_start + n[:, None]) * 4 * 128
            + kv_head * 128
            + d[None, :],
            mask=n[:, None] < k_len,
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)) * SM_SCALE
        causal_limit = (m - q_start) + 1 + k_len - q_len
        valid = (
            valid_m[:, None]
            & (n[None, :] < k_len)
            & (n[None, :] < causal_limit[:, None])
        )
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=1)
        block_max = tl.where(valid_m, block_max, 0.0)
        new_max = tl.maximum(running_max, block_max)
        alpha = tl.exp(running_max - new_max)
        p = tl.exp(scores - new_max[:, None])
        normalizer = normalizer * alpha + tl.sum(p, axis=1)
        vv = tl.load(
            v_ptr
            + (k_start + n[:, None]) * 4 * 128
            + kv_head * 128
            + d[None, :],
            mask=n[:, None] < k_len,
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vv)
        running_max = new_max
        start_n += BLOCK_N

    acc = acc / normalizer[:, None]
    tl.store(
        o_ptr + m[:, None] * 32 * 128 + q_head * 128 + d[None, :],
        acc,
        mask=valid_m[:, None],
    )
    lse = (running_max + tl.log(normalizer)) * 1.4426950408889634
    tl.store(lse_ptr + m * 32 + q_head, lse, mask=valid_m)


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    batch = qo_indptr.shape[0] - 1
    if batch == 1 and q.shape[0] <= 64 and k.shape[0] <= 64:
        nq, nk = q.shape[0], k.shape[0]
        output = torch.empty_like(q)
        lse = torch.empty((nq, 32), dtype=torch.float32, device=q.device)
        block_n = triton.next_power_of_2(nk)
        block_n = max(block_n, 16)
        if nq <= 8:
            _tiny_gqa_kernel[(nq, 4)](
                q,
                k,
                v,
                output,
                lse,
                sm_scale,
                NQ=nq,
                NK=nk,
                BLOCK_N=block_n,
                num_warps=4,
            )
        else:
            block_m = 16
            _small_attention_kernel[(triton.cdiv(nq, block_m), 32)](
                q,
                k,
                v,
                output,
                lse,
                sm_scale,
                NQ=nq,
                NK=nk,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=4,
            )
        return output, lse

    if batch > 1 and q.shape[0] <= 1024:
        nq = q.shape[0]
        output = torch.empty_like(q)
        lse = torch.empty((nq, 32), dtype=torch.float32, device=q.device)
        block_m = 16
        grid_blocks = triton.cdiv(nq, block_m) + batch
        _small_ragged_kernel[(grid_blocks, 32)](
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            output,
            lse,
            sm_scale,
            BATCH=batch,
            BLOCK_M=block_m,
            BLOCK_N=64,
            num_warps=4,
        )
        return output, lse

    output, lse_head_major = aiter.flash_attn_varlen_func(
        q,
        k,
        v,
        qo_indptr,
        kv_indptr,
        q.shape[0],
        k.shape[0],
        softmax_scale=sm_scale,
        causal=True,
        return_lse=True,
        how_v3_bf16_cvt=0,
    )
    # FlashAttention returns natural-log LSE as [heads, total_q].  Convert in
    # place, then expose the required [total_q, heads] layout as a view.
    lse_head_major.mul_(_LOG2E)
    return output, lse_head_major.transpose(0, 1)
