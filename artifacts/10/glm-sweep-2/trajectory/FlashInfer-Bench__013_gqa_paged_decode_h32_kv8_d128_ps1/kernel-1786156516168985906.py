import torch
import math
import triton
import triton.language as tl

INV_LOG2 = 1.0 / math.log(2.0)


@triton.jit
def _decode_attn_kernel(
    q_ptr,          # [B, 32, 128] bf16
    k_ptr,          # [num_pages, 8, 128] bf16  (k_cache squeezed)
    v_ptr,          # [num_pages, 8, 128] bf16
    kv_indices_ptr, # [total_kv_indices] int32
    kv_indptr_ptr,  # [B+1] int32
    out_ptr,        # [B, 32, 128] bf16
    lse_ptr,        # [B, 32] f32
    sm_scale,
    inv_log2,
    GQA_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_kv_heads = tl.num_programs(1)
    b = pid // n_kv_heads
    kv_head = pid % n_kv_heads

    seq_start = tl.load(kv_indptr_ptr + b).to(tl.int64)
    seq_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int64)
    seq_len = seq_end - seq_start

    # q for this kv_head: GQA_RATIO query heads, each HEAD_DIM dims
    q_head_base = b * 32 + kv_head * GQA_RATIO
    dim_offsets = tl.arange(0, HEAD_DIM)  # [128]

    q_offs = (q_head_base + tl.arange(0, GQA_RATIO)[:, None]) * HEAD_DIM + dim_offsets[None, :]
    q = tl.load(q_ptr + q_offs).to(tl.float32)  # [GQA_RATIO, HEAD_DIM]

    # Online softmax accumulators
    m_i = tl.full([GQA_RATIO], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([GQA_RATIO], dtype=tl.float32)
    acc = tl.zeros([GQA_RATIO, HEAD_DIM], dtype=tl.float32)

    num_blocks = tl.cdiv(seq_len, BLOCK_N)

    for n in range(0, num_blocks):
        start_n = n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_n = offs_n < seq_end

        token_idx = tl.load(kv_indices_ptr + seq_start + offs_n, mask=mask_n, other=0).to(tl.int64)

        # Load K: [BLOCK_N, HEAD_DIM]
        k_offs = token_idx[:, None] * (8 * HEAD_DIM) + kv_head * HEAD_DIM + dim_offsets[None, :]
        k = tl.load(k_ptr + k_offs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # logits = Q @ K^T * scale  -> [GQA_RATIO, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(mask_n[None, :], qk, -float("inf"))

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp(m_i - m_ij)
        p = tl.math.exp(qk - m_ij[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V: [BLOCK_N, HEAD_DIM]
        v_offs = token_idx[:, None] * (8 * HEAD_DIM) + kv_head * HEAD_DIM + dim_offsets[None, :]
        v = tl.load(v_ptr + v_offs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        acc = acc + tl.dot(p, v)
        m_i = m_ij

    # lse = logsumexp = m_final + log(l_i), then convert to 2-based
    lse_val = tl.where(l_i > 0, (m_i + tl.log(l_i)) * inv_log2, -float("inf"))

    # Store output
    out_head_base = b * 32 + kv_head * GQA_RATIO
    out_offs = (out_head_base + tl.arange(0, GQA_RATIO)[:, None]) * HEAD_DIM + dim_offsets[None, :]
    tl.store(out_ptr + out_offs, acc.to(tl.bfloat16))

    # Store LSE
    lse_offs = out_head_base + tl.arange(0, GQA_RATIO)
    tl.store(lse_ptr + lse_offs, lse_val)


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape

    device = q.device

    output = torch.empty(
        (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.empty(
        (batch_size, num_qo_heads), dtype=torch.float32, device=device
    )

    k_flat = k_cache.squeeze(1).contiguous()
    v_flat = v_cache.squeeze(1).contiguous()
    q_contig = q.contiguous()

    BLOCK_N = 64

    grid = (batch_size, num_kv_heads)

    _decode_attn_kernel[grid](
        q_contig, k_flat, v_flat,
        kv_indices, kv_indptr,
        output, lse,
        float(sm_scale), INV_LOG2,
        GQA_RATIO=4,
        HEAD_DIM=128,
        BLOCK_N=BLOCK_N,
    )

    return output, lse
