import math
import torch
import triton
import triton.language as tl


@triton.jit
def _mla_decode_kernel(
    q_nope_ptr, q_pe_ptr, ckv_cache_ptr, kpe_cache_ptr,
    kv_indices_ptr, kv_indptr_ptr, output_ptr, lse_ptr,
    sm_scale,
    stride_qn_b, stride_qn_h, stride_qp_b, stride_qp_h,
    stride_kc_p, stride_kp_p, stride_kvi, stride_indptr,
    stride_out_b, stride_out_h, stride_lse_b,
    num_qo_heads: tl.constexpr, D_CKV: tl.constexpr, D_KPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // num_qo_heads
    head_idx = pid % num_qo_heads
    page_beg = tl.load(kv_indptr_ptr + batch_idx * stride_indptr).to(tl.int32)
    page_end = tl.load(kv_indptr_ptr + (batch_idx + 1) * stride_indptr).to(tl.int32)
    seq_len = page_end - page_beg
    off_ckv = tl.arange(0, D_CKV)
    off_kpe = tl.arange(0, D_KPE)
    qn = tl.load(q_nope_ptr + batch_idx * stride_qn_b + head_idx * stride_qn_h + off_ckv).to(tl.float32)
    qp = tl.load(q_pe_ptr + batch_idx * stride_qp_b + head_idx * stride_qp_h + off_kpe).to(tl.float32)
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D_CKV], dtype=tl.float32)
    sm_scale_log2 = sm_scale * 1.4426950408889634
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        kv_idx = tl.load(kv_indices_ptr + (page_beg + offs_n) * stride_kvi, mask=n_mask, other=0).to(tl.int32)
        kptr_ckv = kv_idx[:, None] * stride_kc_p + off_ckv[None, :]
        kptr_kpe = kv_idx[:, None] * stride_kp_p + off_kpe[None, :]
        kc = tl.load(ckv_cache_ptr + kptr_ckv, mask=n_mask[:, None], other=0.0).to(tl.float32)
        kp = tl.load(kpe_cache_ptr + kptr_kpe, mask=n_mask[:, None], other=0.0).to(tl.float32)
        qk = tl.sum(qn[None, :] * kc, axis=1) + tl.sum(qp[None, :] * kp, axis=1)
        qk = qk * sm_scale_log2
        qk = tl.where(n_mask, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new)
        p = tl.where(n_mask, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * kc, axis=0)
        m_i = m_new
    lse_val = m_i + tl.math.log2(l_i)
    out = acc / l_i
    tl.store(output_ptr + batch_idx * stride_out_b + head_idx * stride_out_h + off_ckv, out.to(tl.bfloat16))
    tl.store(lse_ptr + batch_idx * stride_lse_b + head_idx, lse_val)


@triton.jit
def _mla_decode_split_kernel(
    q_nope_ptr, q_pe_ptr, ckv_cache_ptr, kpe_cache_ptr,
    kv_indices_ptr, kv_indptr_ptr,
    partial_m_ptr,     # [B, H, n_splits] float32  (running max)
    partial_l_ptr,     # [B, H, n_splits] float32  (running sum)
    partial_out_ptr,   # [B, H, n_splits, D_CKV] float32  (unnormalized acc)
    sm_scale,
    stride_qn_b, stride_qn_h, stride_qp_b, stride_qp_h,
    stride_kc_p, stride_kp_p, stride_kvi, stride_indptr,
    stride_pm_b, stride_pm_h, stride_pm_s,
    stride_pl_b, stride_pl_h, stride_pl_s,
    stride_pout_b, stride_pout_h, stride_pout_s,
    num_qo_heads: tl.constexpr, D_CKV: tl.constexpr, D_KPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    split_idx = tl.program_id(1)
    n_splits_total = tl.num_programs(1)

    batch_idx = pid // num_qo_heads
    head_idx = pid % num_qo_heads

    page_beg = tl.load(kv_indptr_ptr + batch_idx * stride_indptr).to(tl.int32)
    page_end = tl.load(kv_indptr_ptr + (batch_idx + 1) * stride_indptr).to(tl.int32)
    seq_len = page_end - page_beg

    chunk_size = (seq_len + n_splits_total - 1) // n_splits_total
    split_beg = split_idx * chunk_size
    split_end = tl.minimum(split_beg + chunk_size, seq_len)
    split_len = split_end - split_beg

    off_ckv = tl.arange(0, D_CKV)
    off_kpe = tl.arange(0, D_KPE)

    if split_len <= 0:
        tl.store(partial_m_ptr + batch_idx * stride_pm_b + head_idx * stride_pm_h + split_idx * stride_pm_s,
                 -float("inf"))
        tl.store(partial_l_ptr + batch_idx * stride_pl_b + head_idx * stride_pl_h + split_idx * stride_pl_s,
                 0.0)
        tl.store(partial_out_ptr + batch_idx * stride_pout_b + head_idx * stride_pout_h + split_idx * stride_pout_s + off_ckv,
                 tl.zeros([D_CKV], dtype=tl.float32))
    else:
        qn = tl.load(q_nope_ptr + batch_idx * stride_qn_b + head_idx * stride_qn_h + off_ckv).to(tl.float32)
        qp = tl.load(q_pe_ptr + batch_idx * stride_qp_b + head_idx * stride_qp_h + off_kpe).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([D_CKV], dtype=tl.float32)
        sm_scale_log2 = sm_scale * 1.4426950408889634

        for start_n in range(0, split_len, BLOCK_N):
            offs_n = split_beg + start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_end
            kv_idx = tl.load(kv_indices_ptr + (page_beg + offs_n) * stride_kvi, mask=n_mask, other=0).to(tl.int32)
            kptr_ckv = kv_idx[:, None] * stride_kc_p + off_ckv[None, :]
            kptr_kpe = kv_idx[:, None] * stride_kp_p + off_kpe[None, :]
            kc = tl.load(ckv_cache_ptr + kptr_ckv, mask=n_mask[:, None], other=0.0).to(tl.float32)
            kp = tl.load(kpe_cache_ptr + kptr_kpe, mask=n_mask[:, None], other=0.0).to(tl.float32)
            qk = tl.sum(qn[None, :] * kc, axis=1) + tl.sum(qp[None, :] * kp, axis=1)
            qk = qk * sm_scale_log2
            qk = tl.where(n_mask, qk, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(qk, axis=0))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new)
            p = tl.where(n_mask, p, 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * kc, axis=0)
            m_i = m_new

        tl.store(partial_m_ptr + batch_idx * stride_pm_b + head_idx * stride_pm_h + split_idx * stride_pm_s, m_i)
        tl.store(partial_l_ptr + batch_idx * stride_pl_b + head_idx * stride_pl_h + split_idx * stride_pl_s, l_i)
        tl.store(partial_out_ptr + batch_idx * stride_pout_b + head_idx * stride_pout_h + split_idx * stride_pout_s + off_ckv, acc)


@triton.jit
def _mla_merge_kernel(
    partial_m_ptr, partial_l_ptr, partial_out_ptr,
    output_ptr, lse_ptr,
    stride_pm_b, stride_pm_h, stride_pm_s,
    stride_pl_b, stride_pl_h, stride_pl_s,
    stride_pout_b, stride_pout_h, stride_pout_s,
    stride_out_b, stride_out_h, stride_lse_b,
    D_CKV: tl.constexpr, N_SPLITS: tl.constexpr, BLOCK_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // 16
    head_idx = pid % 16
    off_ckv = tl.arange(0, D_CKV)
    off_splits = tl.arange(0, BLOCK_SPLITS)
    split_mask = off_splits < N_SPLITS

    p_m = tl.load(partial_m_ptr + batch_idx * stride_pm_b + head_idx * stride_pm_h + off_splits * stride_pm_s,
                  mask=split_mask, other=-float("inf"))
    m_global = tl.max(p_m, axis=0)
    alpha = tl.math.exp2(p_m - m_global)
    alpha = tl.where(split_mask, alpha, 0.0)

    p_l = tl.load(partial_l_ptr + batch_idx * stride_pl_b + head_idx * stride_pl_h + off_splits * stride_pl_s,
                  mask=split_mask, other=0.0)
    l_weighted = p_l * alpha
    l_global = tl.sum(l_weighted, axis=0)

    p_out = tl.load(partial_out_ptr + batch_idx * stride_pout_b + head_idx * stride_pout_h + off_splits[:, None] * stride_pout_s + off_ckv[None, :],
                    mask=split_mask[:, None], other=0.0)
    out = tl.sum(alpha[:, None] * p_out, axis=0) / l_global

    lse_val = m_global + tl.math.log2(l_global)
    tl.store(output_ptr + batch_idx * stride_out_b + head_idx * stride_out_h + off_ckv, out.to(tl.bfloat16))
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

    ckv = ckv_cache[:, 0, :].contiguous()
    kpe = kpe_cache[:, 0, :].contiguous()
    qn = q_nope.contiguous()
    qp = q_pe.contiguous()
    kv_indices_i32 = kv_indices.to(torch.int32).contiguous()
    kv_indptr_i32 = kv_indptr.to(torch.int32).contiguous()

    output = torch.empty(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.empty(
        (batch_size, num_qo_heads), dtype=torch.float32, device=device
    )

    total_programs = batch_size * num_qo_heads
    num_cus = 256

    n_splits = max(1, num_cus // total_programs)
    n_splits = min(n_splits, 16)

    BLOCK_N = 128

    if n_splits <= 1:
        grid = (total_programs,)
        _mla_decode_kernel[grid](
            qn, qp, ckv, kpe,
            kv_indices_i32, kv_indptr_i32,
            output, lse,
            sm_scale,
            qn.stride(0), qn.stride(1), qp.stride(0), qp.stride(1),
            ckv.stride(0), kpe.stride(0),
            kv_indices_i32.stride(0), kv_indptr_i32.stride(0),
            output.stride(0), output.stride(1), lse.stride(0),
            num_qo_heads=num_qo_heads, D_CKV=head_dim_ckv, D_KPE=head_dim_kpe,
            BLOCK_N=BLOCK_N, num_warps=4, num_stages=2,
        )
    else:
        partial_m = torch.empty(
            (batch_size, num_qo_heads, n_splits), dtype=torch.float32, device=device
        )
        partial_l = torch.empty(
            (batch_size, num_qo_heads, n_splits), dtype=torch.float32, device=device
        )
        partial_out = torch.empty(
            (batch_size, num_qo_heads, n_splits, head_dim_ckv), dtype=torch.float32, device=device
        )

        grid = (total_programs, n_splits)
        _mla_decode_split_kernel[grid](
            qn, qp, ckv, kpe,
            kv_indices_i32, kv_indptr_i32,
            partial_m, partial_l, partial_out,
            sm_scale,
            qn.stride(0), qn.stride(1), qp.stride(0), qp.stride(1),
            ckv.stride(0), kpe.stride(0),
            kv_indices_i32.stride(0), kv_indptr_i32.stride(0),
            partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
            partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
            partial_out.stride(0), partial_out.stride(1), partial_out.stride(2),
            num_qo_heads=num_qo_heads, D_CKV=head_dim_ckv, D_KPE=head_dim_kpe,
            BLOCK_N=BLOCK_N, num_warps=4, num_stages=2,
        )

        BLOCK_SPLITS = triton.next_power_of_2(n_splits)
        merge_grid = (total_programs,)
        _mla_merge_kernel[merge_grid](
            partial_m, partial_l, partial_out,
            output, lse,
            partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
            partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
            partial_out.stride(0), partial_out.stride(1), partial_out.stride(2),
            output.stride(0), output.stride(1), lse.stride(0),
            D_CKV=head_dim_ckv, N_SPLITS=n_splits, BLOCK_SPLITS=BLOCK_SPLITS,
            num_warps=4, num_stages=1,
        )

    return {"output": output, "lse": lse}
