import torch
import triton
import triton.language as tl


@triton.jit
def _mla_prefill(
    q_nope, q_pe, ckv, kpe, qo_indptr, kv_indptr, kv_indices,
    output, lse, sm_scale: tl.float32,
    NUM_BATCHES: tl.constexpr, BLOCK_B: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # The launch is over the sum of padded per-sequence query blocks.  Derive
    # its ragged (batch, local block) coordinate without a CPU synchronization.
    offs_b = tl.arange(0, BLOCK_B)
    valid_b = offs_b < NUM_BATCHES
    q_starts = tl.load(qo_indptr + offs_b, mask=valid_b, other=0)
    q_ends = tl.load(qo_indptr + offs_b + 1, mask=valid_b, other=0)
    q_lens = tl.maximum(q_ends - q_starts, 0)
    blocks = (q_lens + BLOCK_M - 1) // BLOCK_M
    block_ends = tl.cumsum(blocks, axis=0)
    block_starts = block_ends - blocks
    is_ours = valid_b & (block_id >= block_starts) & (block_id < block_ends)
    active = tl.sum(is_ours, axis=0)
    if active == 0:
        return
    batch_id = tl.sum(tl.where(is_ours, offs_b, 0), axis=0)
    local_block = block_id - tl.sum(
        tl.where(is_ours, block_starts, 0), axis=0
    )

    q_start = tl.load(qo_indptr + batch_id)
    q_end = tl.load(qo_indptr + batch_id + 1)
    page_start = tl.load(kv_indptr + batch_id)
    page_end = tl.load(kv_indptr + batch_id + 1)
    q_len = q_end - q_start
    kv_len = page_end - page_start
    prefix_len = kv_len - q_len

    offs_m = tl.arange(0, BLOCK_M)
    local_m = local_block * BLOCK_M + offs_m
    rows = q_start + local_m
    mask_m = rows < q_end
    offs_d = tl.arange(0, 512)
    offs_p = tl.arange(0, 64)

    if kv_len <= 0:
        tl.store(
            output + (rows[:, None] * 16 + head_id) * 512 + offs_d[None, :],
            0.0,
            mask=mask_m[:, None],
        )
        tl.store(lse + rows * 16 + head_id, -float("inf"), mask=mask_m)
        return

    qn = tl.load(
        q_nope + (rows[:, None] * 16 + head_id) * 512 + offs_d[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )
    qp = tl.load(
        q_pe + (rows[:, None] * 16 + head_id) * 64 + offs_p[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )

    # Online softmax.  Invalid rows use m=0 so an all-masked padded row never
    # forms (-inf)-(-inf), even though its eventual stores are masked out.
    m_i = tl.where(mask_m, -float("inf"), 0.0)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 512), tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    stop_n = tl.minimum(kv_len, prefix_len + (local_block + 1) * BLOCK_M)

    for start_n in tl.range(0, stop_n, BLOCK_N):
        pos_n = start_n + offs_n
        mask_n = pos_n < kv_len
        pages = tl.load(
            kv_indices + page_start + pos_n, mask=mask_n, other=0
        )
        kc = tl.load(
            ckv + pages[:, None] * 512 + offs_d[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        kp = tl.load(
            kpe + pages[:, None] * 64 + offs_p[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )

        scores = tl.dot(qn, tl.trans(kc), out_dtype=tl.float32)
        scores += tl.dot(qp, tl.trans(kp), out_dtype=tl.float32)
        scores *= sm_scale
        causal = pos_n[None, :] <= (prefix_len + local_m[:, None])
        score_mask = mask_m[:, None] & mask_n[None, :] & causal
        scores = tl.where(score_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(probs, axis=1)
        acc *= alpha[:, None]
        acc = tl.dot(probs.to(tl.bfloat16), kc, acc=acc)
        m_i = m_new

    result = acc / l_i[:, None]
    out_ptrs = output + (rows[:, None] * 16 + head_id) * 512 + offs_d[None, :]
    tl.store(out_ptrs, result, mask=mask_m[:, None])
    # Reference returns natural logsumexp divided by ln(2).
    lse_value = m_i * 1.4426950408889634 + tl.log2(l_i)
    tl.store(lse + rows * 16 + head_id, lse_value, mask=mask_m)


@triton.jit
def _mla_prefill_flat(
    q_nope, q_pe, ckv, kpe, qo_indptr, kv_indptr, kv_indices,
    output, lse, sm_scale: tl.float32,
    NUM_BATCHES, BLOCK_B: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr,
):
    block_id = tl.program_id(0)

    offs_b = tl.arange(0, BLOCK_B)
    valid_b = offs_b < NUM_BATCHES
    q_starts = tl.load(qo_indptr + offs_b, mask=valid_b, other=0)
    q_ends = tl.load(qo_indptr + offs_b + 1, mask=valid_b, other=0)
    q_lens = tl.maximum(q_ends - q_starts, 0)
    # Flatten [query, head] so every matrix row is useful even for q_len=1.
    blocks = (q_lens * 16 + BLOCK_R - 1) // BLOCK_R
    block_ends = tl.cumsum(blocks, axis=0)
    block_starts = block_ends - blocks
    is_ours = valid_b & (block_id >= block_starts) & (block_id < block_ends)
    active = tl.sum(is_ours, axis=0)
    if active == 0:
        return
    batch_id = tl.sum(tl.where(is_ours, offs_b, 0), axis=0)
    local_block = block_id - tl.sum(
        tl.where(is_ours, block_starts, 0), axis=0
    )

    q_start = tl.load(qo_indptr + batch_id)
    q_end = tl.load(qo_indptr + batch_id + 1)
    page_start = tl.load(kv_indptr + batch_id)
    page_end = tl.load(kv_indptr + batch_id + 1)
    q_len = q_end - q_start
    kv_len = page_end - page_start
    prefix_len = kv_len - q_len

    offs_r = tl.arange(0, BLOCK_R)
    flat_r = local_block * BLOCK_R + offs_r
    local_q = flat_r // 16
    heads = flat_r % 16
    rows = q_start + local_q
    mask_r = rows < q_end
    offs_d = tl.arange(0, 512)
    offs_p = tl.arange(0, 64)

    if kv_len <= 0:
        tl.store(
            output + (rows[:, None] * 16 + heads[:, None]) * 512 + offs_d[None, :],
            0.0, mask=mask_r[:, None],
        )
        tl.store(
            lse + rows * 16 + heads, -float("inf"), mask=mask_r
        )
        return

    qn = tl.load(
        q_nope + (rows[:, None] * 16 + heads[:, None]) * 512 + offs_d[None, :],
        mask=mask_r[:, None], other=0.0,
    )
    qp = tl.load(
        q_pe + (rows[:, None] * 16 + heads[:, None]) * 64 + offs_p[None, :],
        mask=mask_r[:, None], other=0.0,
    )

    m_i = tl.where(mask_r, -float("inf"), 0.0)
    l_i = tl.zeros((BLOCK_R,), tl.float32)
    acc = tl.zeros((BLOCK_R, 512), tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    queries_in_block = (local_block + 1) * (BLOCK_R // 16)
    stop_n = tl.minimum(kv_len, prefix_len + tl.minimum(q_len, queries_in_block))

    for start_n in tl.range(0, stop_n, BLOCK_N):
        pos_n = start_n + offs_n
        mask_n = pos_n < kv_len
        pages = tl.load(
            kv_indices + page_start + pos_n, mask=mask_n, other=0
        )
        kc = tl.load(
            ckv + pages[:, None] * 512 + offs_d[None, :],
            mask=mask_n[:, None], other=0.0,
        )
        kp = tl.load(
            kpe + pages[:, None] * 64 + offs_p[None, :],
            mask=mask_n[:, None], other=0.0,
        )
        scores = tl.dot(qn, tl.trans(kc), out_dtype=tl.float32)
        scores += tl.dot(qp, tl.trans(kp), out_dtype=tl.float32)
        scores *= sm_scale
        causal = pos_n[None, :] <= (prefix_len + local_q[:, None])
        scores = tl.where(
            mask_r[:, None] & mask_n[None, :] & causal,
            scores, -float("inf"),
        )
        block_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(probs, axis=1)
        acc *= alpha[:, None]
        acc = tl.dot(probs.to(tl.bfloat16), kc, acc=acc)
        m_i = m_new

    result = acc / l_i[:, None]
    tl.store(
        output + (rows[:, None] * 16 + heads[:, None]) * 512 + offs_d[None, :],
        result, mask=mask_r[:, None],
    )
    lse_value = m_i * 1.4426950408889634 + tl.log2(l_i)
    tl.store(lse + rows * 16 + heads, lse_value, mask=mask_r)


@triton.jit
def _mla_lse_flat(
    q_nope, q_pe, ckv, kpe, qo_indptr, kv_indptr, kv_indices,
    lse, sm_scale: tl.float32, NUM_BATCHES,
    BLOCK_B: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr,
):
    block_id = tl.program_id(0)
    offs_b = tl.arange(0, BLOCK_B)
    valid_b = offs_b < NUM_BATCHES
    q_starts = tl.load(qo_indptr + offs_b, mask=valid_b, other=0)
    q_ends = tl.load(qo_indptr + offs_b + 1, mask=valid_b, other=0)
    q_lens = tl.maximum(q_ends - q_starts, 0)
    blocks = (q_lens * 16 + BLOCK_R - 1) // BLOCK_R
    block_ends = tl.cumsum(blocks, axis=0)
    block_starts = block_ends - blocks
    is_ours = valid_b & (block_id >= block_starts) & (block_id < block_ends)
    if tl.sum(is_ours, axis=0) == 0:
        return
    batch_id = tl.sum(tl.where(is_ours, offs_b, 0), axis=0)
    local_block = block_id - tl.sum(
        tl.where(is_ours, block_starts, 0), axis=0
    )

    q_start = tl.load(qo_indptr + batch_id)
    q_end = tl.load(qo_indptr + batch_id + 1)
    page_start = tl.load(kv_indptr + batch_id)
    page_end = tl.load(kv_indptr + batch_id + 1)
    q_len = q_end - q_start
    kv_len = page_end - page_start
    prefix_len = kv_len - q_len
    offs_r = tl.arange(0, BLOCK_R)
    flat_r = local_block * BLOCK_R + offs_r
    local_q = flat_r // 16
    heads = flat_r % 16
    rows = q_start + local_q
    mask_r = rows < q_end
    offs_d = tl.arange(0, 512)
    offs_p = tl.arange(0, 64)

    if kv_len <= 0:
        tl.store(lse + rows * 16 + heads, -float("inf"), mask=mask_r)
        return
    qn = tl.load(
        q_nope + (rows[:, None] * 16 + heads[:, None]) * 512 + offs_d[None, :],
        mask=mask_r[:, None], other=0.0,
    )
    qp = tl.load(
        q_pe + (rows[:, None] * 16 + heads[:, None]) * 64 + offs_p[None, :],
        mask=mask_r[:, None], other=0.0,
    )
    m_i = tl.where(mask_r, -float("inf"), 0.0)
    l_i = tl.zeros((BLOCK_R,), tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    queries_in_block = (local_block + 1) * (BLOCK_R // 16)
    stop_n = tl.minimum(kv_len, prefix_len + tl.minimum(q_len, queries_in_block))
    for start_n in tl.range(0, stop_n, BLOCK_N):
        pos_n = start_n + offs_n
        mask_n = pos_n < kv_len
        pages = tl.load(kv_indices + page_start + pos_n, mask=mask_n, other=0)
        kc = tl.load(
            ckv + pages[:, None] * 512 + offs_d[None, :],
            mask=mask_n[:, None], other=0.0,
        )
        kp = tl.load(
            kpe + pages[:, None] * 64 + offs_p[None, :],
            mask=mask_n[:, None], other=0.0,
        )
        scores = tl.dot(qn, tl.trans(kc), out_dtype=tl.float32)
        scores += tl.dot(qp, tl.trans(kp), out_dtype=tl.float32)
        scores *= sm_scale
        scores = tl.where(
            mask_r[:, None] & mask_n[None, :]
            & (pos_n[None, :] <= prefix_len + local_q[:, None]),
            scores, -float("inf"),
        )
        block_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(probs, axis=1)
        m_i = m_new
    tl.store(
        lse + rows * 16 + heads,
        m_i * 1.4426950408889634 + tl.log2(l_i), mask=mask_r,
    )


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr,
        kv_indices, sm_scale):
    total_q = q_nope.shape[0]
    num_batches = qo_indptr.shape[0] - 1
    output = torch.empty_like(q_nope)
    lse = torch.empty(
        (total_q, 16), dtype=torch.float32, device=q_nope.device
    )

    avg_q = triton.cdiv(total_q, num_batches)
    avg_kv = triton.cdiv(kv_indices.shape[0], num_batches)
    if total_q < 64 or avg_kv > 512 and total_q < 512:
        block_r, block_n, num_warps = 16, 64, 4
    elif total_q < 512:
        block_r, block_n, num_warps = 16, 32, 4
    elif total_q < 2048 or avg_q < 384:
        block_r, block_n, num_warps = 32, 16, 4
    else:
        block_r, block_n, num_warps = 64, 32, 4

    queries_per_block = block_r // 16
    # sum_b ceil(q_len[b] / Q) <= ceil(total_q / Q) + B - 1.
    num_blocks = triton.cdiv(total_q, queries_per_block) + num_batches
    _mla_prefill_flat[(num_blocks,)](
        q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr,
        kv_indices, output, lse, sm_scale,
        NUM_BATCHES=num_batches, BLOCK_B=64,
        BLOCK_R=block_r, BLOCK_N=block_n,
        num_warps=num_warps, num_stages=1,
    )
    return output, lse
