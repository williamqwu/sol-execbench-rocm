import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


# --------------------------------------------------------------------------
# GEMM: C_f32 = A_bf16 @ B_bf16   (fp32 output, fp32 accumulate)
# --------------------------------------------------------------------------
@triton.jit
def _gemm_f32out(A, B, C, M, N, K,
                 sam, sak, sbk, sbn, scm,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 GROUP: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    ngrp = GROUP * nn
    gid = pid // ngrp
    first = gid * GROUP
    gsize = min(nm - first, GROUP)
    pm = first + ((pid % ngrp) % gsize)
    pn = (pid % ngrp) // gsize

    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mm = rm < M
    mn = rn < N
    rm = tl.where(mm, rm, 0)
    rn = tl.where(mn, rn, 0)
    rk = tl.arange(0, BK)

    ap = A + rm[:, None] * sam + rk[None, :] * sak
    bp = B + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros([BM, BN], tl.float32)
    for k0 in range(0, tl.cdiv(K, BK)):
        kmask = rk[None, :] < K - k0 * BK
        a = tl.load(ap, mask=kmask, other=0.0)
        b = tl.load(bp, mask=(rk[:, None] < K - k0 * BK), other=0.0)
        acc = tl.dot(a, b, acc)
        ap += BK * sak
        bp += BK * sbk
    tl.store(C + rm[:, None] * scm + rn[None, :], acc,
             mask=mm[:, None] & mn[None, :])


# --------------------------------------------------------------------------
# Split-precision GEMM for grad_hidden_states.
#   G is fp32 (N, 3*QD). Split each tile into bf16 hi + bf16 lo and do two
#   dots, recovering ~fp32 accuracy of the A operand at 2x bf16 GEMM cost.
#   K axis walks q_weight, then k_weight, then v_weight (each QD x D).
# --------------------------------------------------------------------------
@triton.jit
def _split_block(G, W, acc, rm, rn, woff, QD, sgm, swk,
                 BK: tl.constexpr):
    """acc += split-precision (G[:, woff:woff+QD] @ W)."""
    rk = tl.arange(0, BK)
    gp = G + rm[:, None] * sgm + (woff + rk[None, :])
    bp = W + rk[:, None] * swk + rn[None, :]
    for _ in range(0, tl.cdiv(QD, BK)):
        g = tl.load(gp)
        b = tl.load(bp)
        hi = g.to(tl.bfloat16)
        lo = (g - hi.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(hi, b, acc)
        acc = tl.dot(lo, b, acc)
        gp += BK
        bp += BK * swk
    return acc


@triton.jit
def _gemm_hidden_split(G, WQ, WK, WV, C, M, D, QD,
                       sgm, swk, scm,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       GROUP: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(D, BN)
    ngrp = GROUP * nn
    gid = pid // ngrp
    first = gid * GROUP
    gsize = min(nm - first, GROUP)
    pm = first + ((pid % ngrp) % gsize)
    pn = (pid % ngrp) // gsize

    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mm = rm < M
    mn = rn < D
    rm = tl.where(mm, rm, 0)
    rn = tl.where(mn, rn, 0)
    acc = tl.zeros([BM, BN], tl.float32)
    acc = _split_block(G, WQ, acc, rm, rn, 0, QD, sgm, swk, BK=BK)
    acc = _split_block(G, WK, acc, rm, rn, QD, QD, sgm, swk, BK=BK)
    acc = _split_block(G, WV, acc, rm, rn, 2 * QD, QD, sgm, swk, BK=BK)
    tl.store(C + rm[:, None] * scm + rn[None, :], acc.to(C.dtype.element_ty),
             mask=mm[:, None] & mn[None, :])


# --------------------------------------------------------------------------
# Flash attention forward (varlen by chunk). Emits fp32 O, bf16 O and LSE.
# --------------------------------------------------------------------------
@triton.jit
def _attn_fwd(Q, K, V, OF, OB, LSE, CU,
              stride_qh, stride_qm, stride_om, stride_lh,
              n_mblk, qk_scale2, HOFF,
              BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr):
    pid = tl.program_id(0)
    c = pid // n_mblk
    mi = pid % n_mblk
    h = tl.program_id(1)

    start = tl.load(CU + c).to(tl.int32)
    L = tl.load(CU + c + 1).to(tl.int32) - start
    m0 = mi * BM
    if m0 >= L:
        return

    offs_m = m0 + tl.arange(0, BM)
    offs_d = tl.arange(0, HD)
    mmask = offs_m < L

    q = tl.load(Q + h * stride_qh + (start + offs_m)[:, None] * stride_qm + offs_d[None, :],
                mask=mmask[:, None], other=0.0)

    m_i = tl.full([BM], float('-inf'), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, HD], tl.float32)

    for n0 in range(0, L, BN):
        offs_n = n0 + tl.arange(0, BN)
        nmask = offs_n < L
        kk = tl.load(K + h * stride_qh + (start + offs_n)[:, None] * stride_qm + offs_d[None, :],
                     mask=nmask[:, None], other=0.0)
        vv = tl.load(V + h * stride_qh + (start + offs_n)[:, None] * stride_qm + offs_d[None, :],
                     mask=nmask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(kk)) * qk_scale2
        qk = tl.where(nmask[None, :], qk, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vv.dtype), vv)
        m_i = m_new

    out = acc / l_i[:, None]
    ptr = (start + offs_m)[:, None] * stride_om + (HOFF * h + offs_d)[None, :]
    tl.store(OF + ptr, out, mask=mmask[:, None])
    tl.store(OB + ptr, out.to(OB.dtype.element_ty), mask=mmask[:, None])
    tl.store(LSE + h * stride_lh + (start + offs_m), m_i + tl.log2(l_i), mask=mmask)


# --------------------------------------------------------------------------
# Backward: dK / dV  (one program per (chunk, n-block, head))
# --------------------------------------------------------------------------
@triton.jit
def _attn_bwd_dkv(Q, K, V, DO, LSE, DELTA, GF, GB, CU,
                  stride_qh, stride_qm, stride_om, stride_gm, stride_lh,
                  n_nblk, qk_scale2, sm_scale, HOFF, DK_OFF, DV_OFF,
                  BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr):
    pid = tl.program_id(0)
    c = pid // n_nblk
    nj = pid % n_nblk
    h = tl.program_id(1)

    start = tl.load(CU + c).to(tl.int32)
    L = tl.load(CU + c + 1).to(tl.int32) - start
    n0 = nj * BN
    if n0 >= L:
        return

    offs_n = n0 + tl.arange(0, BN)
    offs_d = tl.arange(0, HD)
    nmask = offs_n < L

    k = tl.load(K + h * stride_qh + (start + offs_n)[:, None] * stride_qm + offs_d[None, :],
                mask=nmask[:, None], other=0.0)
    v = tl.load(V + h * stride_qh + (start + offs_n)[:, None] * stride_qm + offs_d[None, :],
                mask=nmask[:, None], other=0.0)
    vt = tl.trans(v)

    dk = tl.zeros([BN, HD], tl.float32)
    dv = tl.zeros([BN, HD], tl.float32)

    for m0 in range(0, L, BM):
        offs_m = m0 + tl.arange(0, BM)
        mmask = offs_m < L
        q = tl.load(Q + h * stride_qh + (start + offs_m)[:, None] * stride_qm + offs_d[None, :],
                    mask=mmask[:, None], other=0.0)
        do = tl.load(DO + (start + offs_m)[:, None] * stride_om + (HOFF * h + offs_d)[None, :],
                     mask=mmask[:, None], other=0.0)
        lse = tl.load(LSE + h * stride_lh + (start + offs_m), mask=mmask, other=0.0)
        de = tl.load(DELTA + h * stride_lh + (start + offs_m), mask=mmask, other=0.0)

        do_hi = do.to(tl.bfloat16)
        do_lo = (do - do_hi.to(tl.float32)).to(tl.bfloat16)

        qk = tl.dot(q, tl.trans(k)) * qk_scale2
        p = tl.exp2(qk - lse[:, None])
        p = tl.where(mmask[:, None] & nmask[None, :], p, 0.0)
        pt = tl.trans(p).to(tl.bfloat16)

        dv = tl.dot(pt, do_hi, dv)
        dv = tl.dot(pt, do_lo, dv)

        dp = tl.dot(do_hi, vt)
        dp = tl.dot(do_lo, vt, dp)

        ds = p * (dp - de[:, None])
        ds_hi = ds.to(tl.bfloat16)
        ds_lo = (ds - ds_hi.to(tl.float32)).to(tl.bfloat16)
        dsh = tl.trans(ds_hi)
        dsl = tl.trans(ds_lo)
        dk = tl.dot(dsh, q, dk)
        dk = tl.dot(dsl, q, dk)

    dk = dk * sm_scale
    pk = (start + offs_n)[:, None] * stride_gm + (DK_OFF + HOFF * h + offs_d)[None, :]
    pv = (start + offs_n)[:, None] * stride_gm + (DV_OFF + HOFF * h + offs_d)[None, :]
    tl.store(GF + pk, dk, mask=nmask[:, None])
    tl.store(GB + pk, dk.to(GB.dtype.element_ty), mask=nmask[:, None])
    tl.store(GF + pv, dv, mask=nmask[:, None])
    tl.store(GB + pv, dv.to(GB.dtype.element_ty), mask=nmask[:, None])


# --------------------------------------------------------------------------
# Backward: dQ
# --------------------------------------------------------------------------
@triton.jit
def _attn_bwd_dq(Q, K, V, DO, LSE, DELTA, GF, GB, CU,
                 stride_qh, stride_qm, stride_om, stride_gm, stride_lh,
                 n_mblk, qk_scale2, sm_scale, HOFF,
                 BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr):
    pid = tl.program_id(0)
    c = pid // n_mblk
    mi = pid % n_mblk
    h = tl.program_id(1)

    start = tl.load(CU + c).to(tl.int32)
    L = tl.load(CU + c + 1).to(tl.int32) - start
    m0 = mi * BM
    if m0 >= L:
        return

    offs_m = m0 + tl.arange(0, BM)
    offs_d = tl.arange(0, HD)
    mmask = offs_m < L

    q = tl.load(Q + h * stride_qh + (start + offs_m)[:, None] * stride_qm + offs_d[None, :],
                mask=mmask[:, None], other=0.0)
    do = tl.load(DO + (start + offs_m)[:, None] * stride_om + (HOFF * h + offs_d)[None, :],
                 mask=mmask[:, None], other=0.0)
    lse = tl.load(LSE + h * stride_lh + (start + offs_m), mask=mmask, other=0.0)
    de = tl.load(DELTA + h * stride_lh + (start + offs_m), mask=mmask, other=0.0)

    do_hi = do.to(tl.bfloat16)
    do_lo = (do - do_hi.to(tl.float32)).to(tl.bfloat16)

    dq = tl.zeros([BM, HD], tl.float32)

    for n0 in range(0, L, BN):
        offs_n = n0 + tl.arange(0, BN)
        nmask = offs_n < L
        k = tl.load(K + h * stride_qh + (start + offs_n)[:, None] * stride_qm + offs_d[None, :],
                    mask=nmask[:, None], other=0.0)
        v = tl.load(V + h * stride_qh + (start + offs_n)[:, None] * stride_qm + offs_d[None, :],
                    mask=nmask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale2
        p = tl.exp2(qk - lse[:, None])
        p = tl.where(nmask[None, :], p, 0.0)
        vt = tl.trans(v)
        dp = tl.dot(do_hi, vt)
        dp = tl.dot(do_lo, vt, dp)
        ds = p * (dp - de[:, None])
        ds_hi = ds.to(tl.bfloat16)
        ds_lo = (ds - ds_hi.to(tl.float32)).to(tl.bfloat16)
        dq = tl.dot(ds_hi, k, dq)
        dq = tl.dot(ds_lo, k, dq)

    dq = dq * sm_scale
    ptr = (start + offs_m)[:, None] * stride_gm + (HOFF * h + offs_d)[None, :]
    tl.store(GF + ptr, dq, mask=mmask[:, None])
    tl.store(GB + ptr, dq.to(GB.dtype.element_ty), mask=mmask[:, None])


# --------------------------------------------------------------------------
# delta[h, m] = sum_d dO[m, h, d] * O[m, h, d]
# --------------------------------------------------------------------------
@triton.jit
def _delta_kernel(DO, O, DELTA, N, stride_om, stride_lh, HOFF,
                  H: tl.constexpr, HD: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, HD)
    mask = offs_m < N
    ptr = offs_m[:, None] * stride_om + (HOFF * h + offs_d)[None, :]
    a = tl.load(DO + ptr, mask=mask[:, None], other=0.0)
    b = tl.load(O + ptr, mask=mask[:, None], other=0.0)
    tl.store(DELTA + h * stride_lh + offs_m, tl.sum(a * b, 1), mask=mask)


def _gemm(a, b, M, N, K):
    """fp32-output bf16 GEMM."""
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    BM, BN, BK = 128, 128, 64
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _gemm_f32out[grid](a, b, c, M, N, K,
                       a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                       BM=BM, BN=BN, BK=BK, GROUP=8, num_warps=8, num_stages=2)
    return c


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    out_weight: torch.Tensor,
):
    N, D = hidden_states.shape
    H = query_states.shape[1]
    HD = query_states.shape[3]
    QD = H * HD
    nc = cu_seqlens.shape[0] - 1
    sm_scale = HD ** -0.5
    qk_scale2 = sm_scale * LOG2E
    dev = hidden_states.device

    q = query_states.reshape(H, N, HD)
    k = key_states.reshape(H, N, HD)
    v = value_states.reshape(H, N, HD)
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    # Step 1: dO in fp32 (the reference keeps this at fp32; rounding it to
    # bf16 is the single largest source of error downstream).
    GA = _gemm(grad_output, out_weight, N, QD, D)

    OF = torch.empty((N, QD), dtype=torch.float32, device=dev)
    OB = torch.empty((N, QD), dtype=torch.bfloat16, device=dev)
    LSE = torch.empty((H, N), dtype=torch.float32, device=dev)

    BM = BN = 64
    n_mblk = triton.cdiv(N, BM)
    n_nblk = triton.cdiv(N, BN)

    _attn_fwd[(nc * n_mblk, H)](
        q, k, v, OF, OB, LSE, cu_seqlens,
        q.stride(0), q.stride(1), OF.stride(0), LSE.stride(0),
        n_mblk, qk_scale2, HD,
        BM=BM, BN=BN, HD=HD, num_warps=4, num_stages=1,
    )

    grad_out_weight = torch.mm(grad_output.t(), OB)
    grad_out_bias = grad_output.sum(dim=0, dtype=torch.float32).to(torch.bfloat16)

    delta = torch.empty((H, N), dtype=torch.float32, device=dev)
    _delta_kernel[(triton.cdiv(N, 128), H)](
        GA, OF, delta, N, GA.stride(0), delta.stride(0), HD,
        H=H, HD=HD, BM=128, num_warps=4,
    )

    GF = torch.empty((N, 3 * QD), dtype=torch.float32, device=dev)
    GB = torch.empty((N, 3 * QD), dtype=torch.bfloat16, device=dev)

    _attn_bwd_dkv[(nc * n_nblk, H)](
        q, k, v, GA, LSE, delta, GF, GB, cu_seqlens,
        q.stride(0), q.stride(1), GA.stride(0), GF.stride(0), LSE.stride(0),
        n_nblk, qk_scale2, sm_scale, HD, QD, 2 * QD,
        BM=BM, BN=BN, HD=HD, num_warps=4, num_stages=1,
    )
    _attn_bwd_dq[(nc * n_mblk, H)](
        q, k, v, GA, LSE, delta, GF, GB, cu_seqlens,
        q.stride(0), q.stride(1), GA.stride(0), GF.stride(0), LSE.stride(0),
        n_mblk, qk_scale2, sm_scale, HD,
        BM=BM, BN=BN, HD=HD, num_warps=4, num_stages=1,
    )

    gw = torch.mm(GB.t(), hidden_states)
    grad_q_weight = gw[0:QD]
    grad_k_weight = gw[QD:2 * QD]
    grad_v_weight = gw[2 * QD:3 * QD]

    gb = GF.sum(dim=0).to(torch.bfloat16)
    grad_q_bias = gb[0:QD]
    grad_k_bias = gb[QD:2 * QD]
    grad_v_bias = gb[2 * QD:3 * QD]

    grad_hidden_states = torch.empty((N, D), dtype=torch.bfloat16, device=dev)
    HBM, HBN, HBK = 128, 128, 64
    hgrid = (triton.cdiv(N, HBM) * triton.cdiv(D, HBN),)
    _gemm_hidden_split[hgrid](
        GF, q_weight, k_weight, v_weight, grad_hidden_states, N, D, QD,
        GF.stride(0), q_weight.stride(0), grad_hidden_states.stride(0),
        BM=HBM, BN=HBN, BK=HBK, GROUP=8, num_warps=8, num_stages=2,
    )

    return (
        grad_hidden_states,
        grad_q_weight,
        grad_q_bias,
        grad_k_weight,
        grad_k_bias,
        grad_v_weight,
        grad_v_bias,
        grad_out_weight,
        grad_out_bias,
    )
