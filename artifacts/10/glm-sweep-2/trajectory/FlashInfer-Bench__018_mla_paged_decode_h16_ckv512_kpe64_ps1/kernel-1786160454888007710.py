import math
import torch
import triton
import triton.language as tl


@triton.jit
def _mla_decode_kernel(
    q_nope_ptr,       # [B, H, D_CKV] bf16
    q_pe_ptr,         # [B, H, D_KPE] bf16
    ckv_cache_ptr,    # [num_pages, D_CKV] bf16
    kpe_cache_ptr,    # [num_pages, D_KPE] bf16
    kv_indices_ptr,   # [num_kv_indices] int32
    kv_indptr_ptr,    # [B+1] int32
    output_ptr,       # [B, H, D_CKV] bf16
    lse_ptr,          # [B, H] float32
    sm_scale,
    stride_qn_b, stride_qn_h,
    stride_qp_b, stride_qp_h,
    stride_kc_p,
    stride_kp_p,
    stride_kvi,
    stride_indptr,
    stride_out_b, stride_out_h,
    stride_lse_b,
    num_qo_heads: tl.constexpr,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // num_qo_heads
    head_idx = pid % num_qo_heads

    # Sequence range for this batch
    page_beg = tl.load(kv_indptr_ptr + batch_idx * stride_indptr).to(tl.int32)
    page_end = tl.load(kv_indptr_ptr + (batch_idx + 1) * stride_indptr).to(tl.int32)
    seq_len = page_end - page_beg

    # Load query (fp32)
    off_ckv = tl.arange(0, D_CKV)
    off_kpe = tl.arange(0, D_KPE)
    qn = tl.load(q_nope_ptr + batch_idx * stride_qn_b + head_idx * stride_qn_h + off_ckv).to(tl.float32)
    qp = tl.load(q_pe_ptr + batch_idx * stride_qp_b + head_idx * stride_qp_h + off_kpe).to(tl.float32)

    # Online softmax / flash attention
    m_i = tl.full([1], -float("inf"), dtype=tl.float32)
    l_i = tl.full([1], 0.0, dtype=tl.float32)
    acc = tl.zeros([D_CKV], dtype=tl.float32)

    sm_scale_log2 = sm_scale * 1.4426950408889634  # log2(e)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len

        # Load KV indices for this block
        kv_idx = tl.load(kv_indices_ptr + (page_beg + offs_n) * stride_kvi,
                         mask=n_mask, other=0).to(tl.int32)

        # Load Kc [BLOCK_N, D_CKV] and Kp [BLOCK_N, D_KPE]
        kptr_ckv = kv_idx[:, None] * stride_kc_p + off_ckv[None, :]
        kptr_kpe = kv_idx[:, None] * stride_kp_p + off_kpe[None, :]
        kc = tl.load(ckv_cache_ptr + kptr_ckv, mask=n_mask[:, None], other=0.0).to(tl.float32)
        kp = tl.load(kpe_cache_ptr + kptr_kpe, mask=n_mask[:, None], other=0.0).to(tl.float32)

        # QK = qn @ kc.T + qp @ kp.T  -> [BLOCK_N]
        qk = tl.sum(qn[None, :] * kc, axis=1) + tl.sum(qp[None, :] * kp, axis=1)
        qk = qk * sm_scale_log2  # scale by sm_scale * log2(e) -> log2-space
        qk = tl.where(n_mask, qk, -float("inf"))

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new)
        p = tl.where(n_mask, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * kc, axis=0)
        m_i = m_new

    # lse in log2 space is just m_i + log2(l_i)
    # But the reference computes logsumexp(logits_scaled) / log(2)
    # = log2(sum(exp2(logits_scaled * log2(e)))) ... wait.
    # Reference: lse = logsumexp(logits * sm_scale) / log(2)
    # = log(sum(exp(logits * sm_scale))) / log(2)
    # = log2(sum(exp(logits * sm_scale)))
    # In our kernel, qk = (qn@kc.T + qp@kp.T) * sm_scale * log2(e)
    # = logits * sm_scale * log2(e)
    # exp2(qk) = exp2(logits * sm_scale * log2(e)) = exp(logits * sm_scale)
    # So sum(exp2(qk)) = sum(exp(logits * sm_scale))
    # log2(sum(exp2(qk))) = log2(sum(exp(logits*sm_scale))) = lse_ref  ✓
    # With online softmax: l_i = sum(exp2(qk - m_i)), m_i = max(qk)
    # sum(exp2(qk)) = exp2(m_i) * l_i
    # log2(sum(exp2(qk))) = m_i + log2(l_i)  ✓
    lse_val = m_i + tl.math.log2(l_i)

    out = acc / l_i

    # Store
    tl.store(output_ptr + batch_idx * stride_out_b + head_idx * stride_out_h + off_ckv,
             out.to(tl.bfloat16))
    tl.store(lse_ptr + batch_idx * stride_lse_b + head_idx, lse_val)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    device = q_nope.device

    ckv = ckv_cache[:, 0, :].contiguous()  # [num_pages, D_CKV] bf16
    kpe = kpe_cache[:, 0, :].contiguous()  # [num_pages, D_KPE] bf16

    qn = q_nope.contiguous()  # [B, H, D_CKV]
    qp = q_pe.contiguous()    # [B, H, D_KPE]
    kv_indices_i32 = kv_indices.to(torch.int32).contiguous()
    kv_indptr_i32 = kv_indptr.to(torch.int32).contiguous()

    output = torch.empty(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.empty(
        (batch_size, num_qo_heads), dtype=torch.float32, device=device
    )

    BLOCK_N = 64

    grid = (batch_size * num_qo_heads,)
    _mla_decode_kernel[grid](
        qn, qp, ckv, kpe,
        kv_indices_i32, kv_indptr_i32,
        output, lse,
        sm_scale,
        qn.stride(0), qn.stride(1),
        qp.stride(0), qp.stride(1),
        ckv.stride(0),
        kpe.stride(0),
        kv_indices_i32.stride(0),
        kv_indptr_i32.stride(0),
        output.stride(0), output.stride(1),
        lse.stride(0),
        num_qo_heads=num_qo_heads,
        D_CKV=head_dim_ckv,
        D_KPE=head_dim_kpe,
        BLOCK_N=BLOCK_N,
    )

    return {"output": output, "lse": lse}
