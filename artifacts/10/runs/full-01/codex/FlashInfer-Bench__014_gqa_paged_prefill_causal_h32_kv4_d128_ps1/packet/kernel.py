import torch
import triton
import triton.language as tl


@triton.jit
def _init_outputs(output, lse, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    tl.store(output + offs, 0.0, mask=mask)
    is_head_start = (offs & 127) == 0
    tl.store(lse + (offs >> 7), -float("inf"), mask=mask & is_head_start)


@triton.jit
def _paged_attention(
    q,
    k_cache,
    v_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    output,
    lse,
    sm_scale,
    B: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    kv_slot = tl.program_id(0)
    head = tl.program_id(1)

    # Find the sequence owning this slot (upper_bound(kv_indptr, kv_slot)-1).
    lo = 0
    hi = B
    for _ in range(SEARCH_STEPS):
        mid = (lo + hi) >> 1
        boundary = tl.load(kv_indptr + mid + 1)
        goes_right = boundary <= kv_slot
        lo = tl.where(goes_right, mid + 1, lo)
        hi = tl.where(goes_right, hi, mid)
    batch = lo

    kv_start = tl.load(kv_indptr + batch)
    kv_end = tl.load(kv_indptr + batch + 1)
    q_start = tl.load(qo_indptr + batch)
    q_end = tl.load(qo_indptr + batch + 1)
    kv_len = kv_end - kv_start
    q_len = q_end - q_start
    local_slot = kv_slot - kv_start

    # A valid causal query corresponds uniquely to one of the first
    # min(q_len, kv_len) local KV slots.
    valid = local_slot < tl.minimum(q_len, kv_len)
    if valid:
        q_local = local_slot + tl.maximum(q_len - kv_len, 0)
        q_token = q_start + q_local
        max_kv = local_slot + 1 + tl.maximum(kv_len - q_len, 0)
        kv_head = head >> 3

        d = tl.arange(0, 128)
        q_vec = tl.load(q + (q_token * 32 + head) * 128 + d).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((128,), tl.float32)
        start_n = 0
        while start_n < max_kv:
            n = start_n + tl.arange(0, BLOCK_N)
            n_mask = n < max_kv
            page = tl.load(kv_indices + kv_start + n, mask=n_mask, other=0)
            cache_base = page * 512 + kv_head * 128

            k = tl.load(
                k_cache + cache_base[None, :] + d[:, None],
                mask=n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(q_vec[:, None] * k, axis=0) * sm_scale
            scores = tl.where(n_mask, scores, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_ij)
            p = tl.exp(scores - m_ij)

            v = tl.load(
                v_cache + cache_base[:, None] + d[None, :],
                mask=n_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_ij
            start_n += BLOCK_N

        out_base = (q_token * 32 + head) * 128
        tl.store(output + out_base + d, acc / l_i)
        tl.store(lse + q_token * 32 + head, (m_i + tl.log(l_i)) * 1.4426950408889634)


@triton.jit
def _paged_attention_grouped(
    q,
    k_cache,
    v_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    output,
    lse,
    sm_scale,
    B: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    kv_slot = tl.program_id(0)
    kv_head = tl.program_id(1)

    lo = 0
    hi = B
    for _ in range(SEARCH_STEPS):
        mid = (lo + hi) >> 1
        boundary = tl.load(kv_indptr + mid + 1)
        goes_right = boundary <= kv_slot
        lo = tl.where(goes_right, mid + 1, lo)
        hi = tl.where(goes_right, hi, mid)
    batch = lo

    kv_start = tl.load(kv_indptr + batch)
    kv_end = tl.load(kv_indptr + batch + 1)
    q_start = tl.load(qo_indptr + batch)
    q_end = tl.load(qo_indptr + batch + 1)
    kv_len = kv_end - kv_start
    q_len = q_end - q_start
    local_slot = kv_slot - kv_start
    valid = local_slot < tl.minimum(q_len, kv_len)

    if valid:
        q_local = local_slot + tl.maximum(q_len - kv_len, 0)
        q_token = q_start + q_local
        max_kv = local_slot + 1 + tl.maximum(kv_len - q_len, 0)

        h = kv_head * 8 + tl.arange(0, 8)
        d = tl.arange(0, 128)
        q_mat = tl.load(q + (q_token * 32 + h[:, None]) * 128 + d[None, :]).to(
            tl.float32
        )

        m_i = tl.full((8,), -float("inf"), tl.float32)
        l_i = tl.zeros((8,), tl.float32)
        acc = tl.zeros((8, 128), tl.float32)
        start_n = 0
        while start_n < max_kv:
            n = start_n + tl.arange(0, BLOCK_N)
            n_mask = n < max_kv
            page = tl.load(kv_indices + kv_start + n, mask=n_mask, other=0)
            cache_base = page * 512 + kv_head * 128
            k = tl.load(
                k_cache + d[:, None] + cache_base[None, :],
                mask=n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(q_mat[:, :, None] * k[None, :, :], axis=1) * sm_scale
            scores = tl.where(n_mask[None, :], scores, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_ij)
            p = tl.exp(scores - m_ij[:, None])
            p = tl.where(n_mask[None, :], p, 0.0)
            v = tl.load(
                v_cache + cache_base[:, None] + d[None, :],
                mask=n_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            acc = acc * alpha[:, None] + tl.sum(
                p[:, :, None] * v[None, :, :], axis=1
            )
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij
            start_n += BLOCK_N

        out_ptr = output + (q_token * 32 + h[:, None]) * 128 + d[None, :]
        tl.store(out_ptr, acc / l_i[:, None])
        tl.store(
            lse + q_token * 32 + h,
            (m_i + tl.log(l_i)) * 1.4426950408889634,
        )


@triton.jit
def _paged_attention_flash(
    q,
    k_cache,
    v_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    output,
    lse,
    sm_scale,
    B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    work_id = tl.program_id(0)
    head = tl.program_id(1)

    # Map a dense tile id to (sequence, local tile).  The host-side grid is an
    # upper bound of ceil(total_kv / BLOCK_M) + B - 1; the short scan compacts
    # ragged sequences without an auxiliary work-queue launch.
    cumulative = 0
    batch = 0
    q_start = 0
    q_len = 0
    kv_start = 0
    kv_len = 0
    valid_len = 0
    local_slot = 0
    for b in range(B):
        qs = tl.load(qo_indptr + b)
        qe = tl.load(qo_indptr + b + 1)
        ks = tl.load(kv_indptr + b)
        ke = tl.load(kv_indptr + b + 1)
        ql = qe - qs
        kl = ke - ks
        vl = tl.minimum(ql, kl)
        blocks = (vl + BLOCK_M - 1) // BLOCK_M
        belongs = (work_id >= cumulative) & (work_id < cumulative + blocks)
        batch = tl.where(belongs, b, batch)
        q_start = tl.where(belongs, qs, q_start)
        q_len = tl.where(belongs, ql, q_len)
        kv_start = tl.where(belongs, ks, kv_start)
        kv_len = tl.where(belongs, kl, kv_len)
        valid_len = tl.where(belongs, vl, valid_len)
        local_slot = tl.where(belongs, (work_id - cumulative) * BLOCK_M, local_slot)
        cumulative += blocks

    if work_id < cumulative:
        m = tl.arange(0, BLOCK_M)
        d = tl.arange(0, 128)
        row_mask = local_slot + m < valid_len
        q_local = local_slot + m + tl.maximum(q_len - kv_len, 0)
        q_token = q_start + q_local
        q_mat = tl.load(
            q + (q_token[:, None] * 32 + head) * 128 + d[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )

        prefix = tl.maximum(kv_len - q_len, 0)
        max_kv = local_slot + m + 1 + prefix
        block_max_kv = tl.minimum(local_slot + BLOCK_M, valid_len) + prefix

        m_i = tl.where(row_mask, -float("inf"), 0.0)
        l_i = tl.where(row_mask, 0.0, 1.0)
        acc = tl.zeros((BLOCK_M, 128), tl.float32)
        start_n = 0
        kv_head = head >> 3
        while start_n < block_max_kv:
            n = start_n + tl.arange(0, BLOCK_N)
            n_mask = n < block_max_kv
            page = tl.load(kv_indices + kv_start + n, mask=n_mask, other=0)
            cache_base = page * 512 + kv_head * 128

            k = tl.load(
                k_cache + d[:, None] + cache_base[None, :],
                mask=n_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(q_mat, k, out_dtype=tl.float32) * sm_scale
            causal = row_mask[:, None] & (n[None, :] < max_kv[:, None])
            scores = tl.where(causal, scores, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_ij)
            p = tl.exp(scores - m_ij[:, None])
            p = tl.where(causal, p, 0.0)

            v = tl.load(
                v_cache + cache_base[:, None] + d[None, :],
                mask=n_mask[:, None],
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(
                p.to(tl.bfloat16), v, out_dtype=tl.float32
            )
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij
            start_n += BLOCK_N

        out = acc / l_i[:, None]
        out_ptr = output + (q_token[:, None] * 32 + head) * 128 + d[None, :]
        tl.store(out_ptr, out, mask=row_mask[:, None])
        tl.store(
            lse + q_token * 32 + head,
            (m_i + tl.log(l_i)) * 1.4426950408889634,
            mask=row_mask,
        )


def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q = q.shape[0]
    batch_size = qo_indptr.numel() - 1
    num_kv = kv_indices.numel()

    output = torch.empty_like(q)
    lse = torch.empty((total_q, 32), device=q.device, dtype=torch.float32)

    n_elements = output.numel()
    _init_outputs[(triton.cdiv(n_elements, 4096),)](
        output, lse, n_elements=n_elements, BLOCK=4096, num_warps=4
    )

    if num_kv:
        search_steps = max(1, batch_size.bit_length())
        if num_kv > 256:
            max_work = triton.cdiv(num_kv, 64) + batch_size - 1
            _paged_attention_flash[(max_work, 32)](
                q,
                k_cache,
                v_cache,
                qo_indptr,
                kv_indptr,
                kv_indices,
                output,
                lse,
                sm_scale,
                B=batch_size,
                BLOCK_M=64,
                BLOCK_N=64,
                num_warps=2,
            )
        else:
            block_n = 64 if batch_size == 1 and num_kv > 8 else 16
            _paged_attention[(num_kv, 32)](
                q,
                k_cache,
                v_cache,
                qo_indptr,
                kv_indptr,
                kv_indices,
                output,
                lse,
                sm_scale,
                B=batch_size,
                SEARCH_STEPS=search_steps,
                BLOCK_N=block_n,
                num_warps=2,
            )

    return output, lse
