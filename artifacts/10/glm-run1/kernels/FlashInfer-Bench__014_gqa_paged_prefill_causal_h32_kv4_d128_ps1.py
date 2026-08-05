import torch
import triton
import triton.language as tl
import math


@triton.jit
def _flash_attn_causal_kernel(
    Q, K, V, O, LSE,
    sm_scale,
    delta, nq, nkv,
    stride_qh, stride_qq, stride_qd,
    stride_kh, stride_kk, stride_kd,
    stride_vh, stride_vk, stride_vd,
    stride_oh, stride_oq, stride_od,
    stride_lseh, stride_lseq,
    GQA_RATIO: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    qhead_id = tl.program_id(0)
    q_blk_id = tl.program_id(1)

    q_start = q_blk_id * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offs < nq

    kv_head = qhead_id // GQA_RATIO
    K_h = K + kv_head * stride_kh
    V_h = V + kv_head * stride_vh

    Q_blk = tl.load(
        Q + qhead_id * stride_qh + q_offs[:, None] * stride_qq + tl.arange(0, HEAD_DIM)[None, :] * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    )

    m_i = tl.full([BLOCK_Q], -float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_Q], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    max_q_in_blk = tl.minimum(q_start + BLOCK_Q - 1, nq - 1)
    max_kv_global = max_q_in_blk + 1 + delta
    max_kv_global = tl.minimum(max_kv_global, nkv)

    for kv_blk in tl.range(0, max_kv_global, BLOCK_KV):
        kv_offs = kv_blk + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offs < nkv

        K_blk = tl.load(
            K_h + kv_offs[:, None] * stride_kk + tl.arange(0, HEAD_DIM)[None, :] * stride_kd,
            mask=kv_mask[:, None],
            other=0.0,
        )
        V_blk = tl.load(
            V_h + kv_offs[:, None] * stride_vk + tl.arange(0, HEAD_DIM)[None, :] * stride_vd,
            mask=kv_mask[:, None],
            other=0.0,
        )

        scores = tl.dot(Q_blk, tl.trans(K_blk)) * sm_scale

        causal = kv_offs[None, :] < (q_offs[:, None] + 1 + delta)
        valid = causal & kv_mask[None, :] & q_mask[:, None]
        scores = tl.where(valid, scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(scores - m_ij[:, None])
        p = tl.where(valid, p, 0.0)
        l_ij = tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), V_blk)
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    final_scale = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    acc = acc * final_scale[:, None]
    lse_val = tl.where(l_i > 0, m_i + tl.log(l_i), -float("inf"))
    row_valid = l_i > 0
    acc = tl.where(row_valid[:, None], acc, 0.0)

    tl.store(
        O + qhead_id * stride_oh + q_offs[:, None] * stride_oq + tl.arange(0, HEAD_DIM)[None, :] * stride_od,
        acc,
        mask=q_mask[:, None],
    )
    tl.store(
        LSE + qhead_id * stride_lseh + q_offs * stride_lseq,
        lse_val,
        mask=q_mask,
    )


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = qo_indptr.shape[0]

    assert num_qo_heads == 32
    assert num_kv_heads == 4
    assert head_dim == 128
    assert page_size == 1
    assert total_q == qo_indptr[-1].item()

    device = q.device
    gqa_ratio = num_qo_heads // num_kv_heads  # 8

    output = torch.empty(
        (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.empty(
        (total_q, num_qo_heads), dtype=torch.float32, device=device
    )

    q_f32 = q.to(torch.float32)
    k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]
    v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]

    qo_indptr_cpu = qo_indptr.to("cpu")
    kv_indptr_cpu = kv_indptr.to("cpu")
    kv_indices_long = kv_indices.to(torch.long)

    log2 = math.log(2.0)
    neg_inf = float("-inf")

    BLOCK_Q = 64
    BLOCK_KV = 64

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr_cpu[b].item())
        q_end = int(qo_indptr_cpu[b + 1].item())
        kv_start = int(kv_indptr_cpu[b].item())
        kv_end = int(kv_indptr_cpu[b + 1].item())

        nq = q_end - q_start
        nkv = kv_end - kv_start
        if nq <= 0 or nkv <= 0:
            output[q_start:q_end] = 0
            lse[q_start:q_end] = neg_inf
            continue

        page_ids = kv_indices_long[kv_start:kv_end]
        k_batch = k_cache_flat[page_ids]  # [nkv, num_kv_heads, head_dim]
        v_batch = v_cache_flat[page_ids]  # [nkv, num_kv_heads, head_dim]
        # contiguous [num_kv_heads, nkv, head_dim]
        k_seq = k_batch.permute(1, 0, 2).contiguous()
        v_seq = v_batch.permute(1, 0, 2).contiguous()
        # Q for this seq: [nq, num_qo_heads, head_dim] -> [num_qo_heads, nq, head_dim]
        q_seq = q_f32[q_start:q_end].permute(1, 0, 2).contiguous()

        o_seq = torch.empty((num_qo_heads, nq, head_dim), dtype=torch.float32, device=device)
        lse_seq = torch.empty((num_qo_heads, nq), dtype=torch.float32, device=device)

        delta = nkv - nq
        n_q_blocks = triton.cdiv(nq, BLOCK_Q)

        _flash_attn_causal_kernel[(num_qo_heads, n_q_blocks)](
            q_seq, k_seq, v_seq, o_seq, lse_seq,
            sm_scale,
            delta, nq, nkv,
            q_seq.stride(0), q_seq.stride(1), q_seq.stride(2),
            k_seq.stride(0), k_seq.stride(1), k_seq.stride(2),
            v_seq.stride(0), v_seq.stride(1), v_seq.stride(2),
            o_seq.stride(0), o_seq.stride(1), o_seq.stride(2),
            lse_seq.stride(0), lse_seq.stride(1),
            GQA_RATIO=gqa_ratio,
            BLOCK_Q=BLOCK_Q,
            BLOCK_KV=BLOCK_KV,
            HEAD_DIM=head_dim,
        )

        # o_seq: [num_qo_heads, nq, head_dim] -> [nq, num_qo_heads, head_dim]
        output[q_start:q_end] = o_seq.permute(1, 0, 2).to(torch.bfloat16)
        # lse_seq: [num_qo_heads, nq] -> [nq, num_qo_heads]; convert natural->base2
        lse[q_start:q_end] = (lse_seq.permute(1, 0) / log2)

    return output, lse
