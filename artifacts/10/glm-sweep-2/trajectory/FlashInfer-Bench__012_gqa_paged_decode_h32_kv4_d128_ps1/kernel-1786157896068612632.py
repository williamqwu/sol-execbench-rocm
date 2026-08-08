import torch
import math
import triton
import triton.language as tl

_NUM_QO_HEADS = 32
_NUM_KV_HEADS = 4
_HEAD_DIM = 128
_GQA_RATIO = _NUM_QO_HEADS // _NUM_KV_HEADS  # 8
_NUM_SPLITS = 16


@triton.jit
def _flash_decode_split_kernel(
    Q_ptr, K_ptr, V_ptr, KV_indptr_ptr, KV_indices_ptr, Sm_scale,
    Out_partial_ptr,   # [num_splits, batch_size, num_qo_heads, head_dim] float32
    LSE_partial_ptr,    # [num_splits, batch_size, num_qo_heads] float32
    q_b_stride, q_h_stride,
    k_p_stride, k_h_stride,
    v_p_stride, v_h_stride,
    op_s_stride, op_b_stride, op_h_stride,
    lp_s_stride, lp_b_stride,
    NUM_QO_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INV_LOG2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    # grid: (batch_size * NUM_KV_HEADS * NUM_SPLITS,)
    pid = tl.program_id(0)
    split_id = pid % NUM_SPLITS
    rest = pid // NUM_SPLITS
    kv_head_id = rest % NUM_KV_HEADS
    batch_id = rest // NUM_KV_HEADS

    seq_start = tl.load(KV_indptr_ptr + batch_id)
    seq_end = tl.load(KV_indptr_ptr + batch_id + 1)
    seq_len = seq_end - seq_start

    # Split the sequence into NUM_SPLITS chunks
    chunk_len = tl.cdiv(seq_len, NUM_SPLITS)
    chunk_start = split_id * chunk_len
    chunk_end = tl.minimum(chunk_start + chunk_len, seq_len)
    chunk_actual = chunk_end - chunk_start

    q_head_base = kv_head_id * GQA_RATIO
    dim_off = tl.arange(0, HEAD_DIM)
    gqa_off = tl.arange(0, GQA_RATIO)
    q_ptrs = Q_ptr + batch_id * q_b_stride + (q_head_base + gqa_off)[:, None] * q_h_stride + dim_off[None, :]
    q = tl.load(q_ptrs).to(tl.float32)

    scale = tl.load(Sm_scale)

    m_i = tl.full([GQA_RATIO], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([GQA_RATIO], dtype=tl.float32)
    acc = tl.zeros([GQA_RATIO, HEAD_DIM], dtype=tl.float32)

    if chunk_actual > 0:
        num_blocks = tl.cdiv(chunk_actual, BLOCK_N)
        for n in range(0, num_blocks):
            n_off = chunk_start + n * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = n_off < chunk_end
            page_ids = tl.load(KV_indices_ptr + seq_start + n_off, mask=mask_n, other=0)

            k_ptrs = K_ptr + page_ids[:, None] * k_p_stride + kv_head_id * k_h_stride + dim_off[None, :]
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
            logits = tl.dot(q, tl.trans(k)) * scale
            logits = tl.where(mask_n[None, :], logits, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(logits, axis=1))
            alpha = tl.math.exp(m_i - m_ij)
            p = tl.math.exp(logits - m_ij[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            v_ptrs = V_ptr + page_ids[:, None] * v_p_stride + kv_head_id * v_h_stride + dim_off[None, :]
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
            acc = acc + tl.dot(p, v)
            m_i = m_ij

    # Write partial results (in natural-log space for LSE)
    # lse_partial in natural log: m_i + log(l_i)
    # For empty chunks, m_i=-inf, l_i=0 -> lse=-inf, acc=0
    lse_partial = m_i + tl.math.log(l_i)  # natural log space

    op_ptrs = (Out_partial_ptr + split_id * op_s_stride + batch_id * op_b_stride
               + (q_head_base + gqa_off)[:, None] * op_h_stride + dim_off[None, :])
    tl.store(op_ptrs, acc)

    lp_ptrs = LSE_partial_ptr + split_id * lp_s_stride + batch_id * lp_b_stride + (q_head_base + gqa_off)
    tl.store(lp_ptrs, lse_partial)


@triton.jit
def _combine_kernel(
    Out_partial_ptr,   # [num_splits, batch_size, num_qo_heads, head_dim] float32
    LSE_partial_ptr,    # [num_splits, batch_size, num_qo_heads] float32
    Out_ptr,            # [batch_size, num_qo_heads, head_dim] bf16
    LSE_ptr,            # [batch_size, num_qo_heads] float32
    op_s_stride, op_b_stride, op_h_stride,
    lp_s_stride, lp_b_stride,
    out_b_stride, out_h_stride,
    lse_b_stride,
    NUM_QO_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    INV_LOG2: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // NUM_QO_HEADS
    head_id = pid % NUM_QO_HEADS

    dim_off = tl.arange(0, HEAD_DIM)

    # Load all partial LSEs: [NUM_SPLITS]
    split_off = tl.arange(0, NUM_SPLITS)
    lse_ptrs = LSE_partial_ptr + split_off * lp_s_stride + batch_id * lp_b_stride + head_id
    lse_parts = tl.load(lse_ptrs)  # [NUM_SPLITS]

    # Online merge of partial results
    m_max = tl.max(lse_parts)
    # If all -inf (empty sequence), output zeros and -inf
    valid = m_max > -float("inf")

    # alpha = exp(lse_parts - m_max)
    alpha = tl.math.exp(lse_parts - m_max)  # [NUM_SPLITS]
    alpha = tl.where(split_off < NUM_SPLITS, alpha, 0.0)
    # Handle -inf entries
    alpha = tl.where(lse_parts > -float("inf"), alpha, 0.0)
    l_sum = tl.sum(alpha)

    # Load partial outputs: [NUM_SPLITS, HEAD_DIM]
    op_ptrs = (Out_partial_ptr + split_off[:, None] * op_s_stride + batch_id * op_b_stride
               + head_id * op_h_stride + dim_off[None, :])
    op = tl.load(op_ptrs)  # [NUM_SPLITS, HEAD_DIM]

    # Weighted sum: acc = sum(alpha[:,None] * op) / l_sum
    acc = tl.sum(alpha[:, None] * op, axis=0)  # [HEAD_DIM]
    out = acc / l_sum

    # LSE (2-based): (m_max + log(l_sum)) / log(2)
    lse_val = (m_max + tl.math.log(l_sum)) * INV_LOG2

    out_ptrs = Out_ptr + batch_id * out_b_stride + head_id * out_h_stride + dim_off
    tl.store(out_ptrs, tl.where(valid, out.to(tl.bfloat16), tl.zeros([HEAD_DIM], dtype=tl.bfloat16)))

    lse_ptrs2 = LSE_ptr + batch_id * lse_b_stride + head_id
    tl.store(lse_ptrs2, tl.where(valid, lse_val, -float("inf")))


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    assert num_qo_heads == 32
    assert num_kv_heads == 4
    assert head_dim == 128
    assert page_size == 1
    assert len_indptr == batch_size + 1

    device = q.device

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    k_flat = k_cache[:, 0].contiguous()
    v_flat = v_cache[:, 0].contiguous()
    sm_scale_t = torch.tensor(float(sm_scale), dtype=torch.float32, device=device)

    num_splits = _NUM_SPLITS

    out_partial = torch.zeros(
        (num_splits, batch_size, num_qo_heads, head_dim), dtype=torch.float32, device=device
    )
    lse_partial = torch.full(
        (num_splits, batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    grid1 = (batch_size * num_kv_heads * num_splits,)
    _flash_decode_split_kernel[grid1](
        q, k_flat, v_flat, kv_indptr, kv_indices, sm_scale_t,
        out_partial, lse_partial,
        q.stride(0), q.stride(1),
        k_flat.stride(0), k_flat.stride(1),
        v_flat.stride(0), v_flat.stride(1),
        out_partial.stride(0), out_partial.stride(1), out_partial.stride(2),
        lse_partial.stride(0), lse_partial.stride(1),
        NUM_QO_HEADS=num_qo_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        GQA_RATIO=_GQA_RATIO,
        BLOCK_N=256,
        INV_LOG2=1.0 / math.log(2.0),
        NUM_SPLITS=num_splits,
    )

    grid2 = (batch_size * num_qo_heads,)
    _combine_kernel[grid2](
        out_partial, lse_partial, output, lse,
        out_partial.stride(0), out_partial.stride(1), out_partial.stride(2),
        lse_partial.stride(0), lse_partial.stride(1),
        output.stride(0), output.stride(1),
        lse.stride(0),
        NUM_QO_HEADS=num_qo_heads,
        HEAD_DIM=head_dim,
        NUM_SPLITS=num_splits,
        INV_LOG2=1.0 / math.log(2.0),
    )

    return output, lse
