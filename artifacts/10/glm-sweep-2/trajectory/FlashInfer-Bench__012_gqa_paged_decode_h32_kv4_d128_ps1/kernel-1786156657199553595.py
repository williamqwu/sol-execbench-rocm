import torch
import math
import triton
import triton.language as tl

_NUM_QO_HEADS = 32
_NUM_KV_HEADS = 4
_HEAD_DIM = 128
_GQA_RATIO = _NUM_QO_HEADS // _NUM_KV_HEADS  # 8
_LOG2 = math.log(2.0)


@triton.jit
def _flash_decode_kernel(
    # Pointers
    Q_ptr,          # [batch_size, num_qo_heads, head_dim] bf16
    K_ptr,          # [num_pages, num_kv_heads, head_dim] bf16  (page_size=1 squeezed)
    V_ptr,          # [num_pages, num_kv_heads, head_dim] bf16
    KV_indptr_ptr,  # [batch_size+1] int32
    KV_indices_ptr, # [num_kv_indices] int32
    Sm_scale,       # scalar float
    Out_ptr,        # [batch_size, num_qo_heads, head_dim] bf16
    LSE_ptr,        # [batch_size, num_qo_heads] float32
    # Strides
    q_b_stride, q_h_stride,
    k_p_stride, k_h_stride,
    v_p_stride, v_h_stride,
    out_b_stride, out_h_stride,
    lse_b_stride,
    # Constants
    NUM_QO_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INV_LOG2: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles one (batch, kv_head) pair.
    # Within a program we process all GQA_RATIO query heads sharing this kv head.
    batch_id = pid // NUM_KV_HEADS
    kv_head_id = pid % NUM_KV_HEADS

    seq_start = tl.load(KV_indptr_ptr + batch_id)
    seq_end = tl.load(KV_indptr_ptr + batch_id + 1)
    seq_len = seq_end - seq_start

    # Load Q for all GQA_RATIO query heads of this kv head.
    # q_head_base = kv_head_id * GQA_RATIO
    q_head_base = kv_head_id * GQA_RATIO

    # Q: [GQA_RATIO, HEAD_DIM]
    dim_off = tl.arange(0, HEAD_DIM)
    gqa_off = tl.arange(0, GQA_RATIO)
    q_ptrs = Q_ptr + batch_id * q_b_stride + (q_head_base + gqa_off)[:, None] * q_h_stride + dim_off[None, :]
    q = tl.load(q_ptrs).to(tl.float32)  # [GQA_RATIO, HEAD_DIM]

    scale = tl.load(Sm_scale)

    # Online softmax accumulators
    m_i = tl.full([GQA_RATIO], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([GQA_RATIO], dtype=tl.float32)
    acc = tl.zeros([GQA_RATIO, HEAD_DIM], dtype=tl.float32)

    num_blocks = tl.cdiv(seq_len, BLOCK_N)
    for n in range(0, num_blocks):
        n_off = n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = n_off < seq_len

        # Load page indices for this block
        page_ids = tl.load(KV_indices_ptr + seq_start + n_off, mask=mask_n, other=0)  # [BLOCK_N]

        # Load K block: [BLOCK_N, HEAD_DIM]
        k_ptrs = K_ptr + page_ids[:, None] * k_p_stride + kv_head_id * k_h_stride + dim_off[None, :]
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)  # [BLOCK_N, HEAD_DIM]

        # logits = Q @ K^T  -> [GQA_RATIO, BLOCK_N]
        logits = tl.dot(q, tl.trans(k)) * scale  # [GQA_RATIO, BLOCK_N]
        logits = tl.where(mask_n[None, :], logits, -float("inf"))

        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(logits, axis=1))
        alpha = tl.math.exp(m_i - m_ij)
        p = tl.math.exp(logits - m_ij[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V block: [BLOCK_N, HEAD_DIM]
        v_ptrs = V_ptr + page_ids[:, None] * v_p_stride + kv_head_id * v_h_stride + dim_off[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)  # [BLOCK_N, HEAD_DIM]

        acc = acc + tl.dot(p, v)
        m_i = m_ij

    # Finalize
    # lse is 2-based: log2(l_i) + m_i / log2
    # logsumexp(x) = m + log(l); 2-based = (m + log(l)) / log(2)
    lse_val = m_i + tl.math.log(l_i)
    lse_val = lse_val * INV_LOG2

    # Write output: [GQA_RATIO, HEAD_DIM] -> bf16
    out = acc / l_i[:, None]
    out_ptrs = Out_ptr + batch_id * out_b_stride + (q_head_base + gqa_off)[:, None] * out_h_stride + dim_off[None, :]
    tl.store(out_ptrs, out.to(tl.bfloat16))

    # Write LSE: [GQA_RATIO]
    lse_ptrs = LSE_ptr + batch_id * lse_b_stride + (q_head_base + gqa_off)
    tl.store(lse_ptrs, lse_val)


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

    # Squeeze page dimension: [num_pages, num_kv_heads, head_dim]
    k_flat = k_cache[:, 0].contiguous()
    v_flat = v_cache[:, 0].contiguous()

    sm_scale_t = torch.tensor(float(sm_scale), dtype=torch.float32, device=device)

    grid = (batch_size * num_kv_heads,)
    _flash_decode_kernel[grid](
        q, k_flat, v_flat, kv_indptr, kv_indices, sm_scale_t,
        output, lse,
        q.stride(0), q.stride(1),
        k_flat.stride(0), k_flat.stride(1),
        v_flat.stride(0), v_flat.stride(1),
        output.stride(0), output.stride(1),
        lse.stride(0),
        NUM_QO_HEADS=num_qo_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        GQA_RATIO=_GQA_RATIO,
        BLOCK_N=64,
        INV_LOG2=1.0 / math.log(2.0),
    )

    return output, lse
