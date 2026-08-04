import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: grad_attn_scores
#
#   dP[m,n]   = sum_d dO[m,d] * V[n,d]                     (fp32 accum)
#   dAW[m,n]  = (dP[m,n] * mask[m,n]) / (1-p)
#   s[m]      = sum_n dAW[m,n] * A[m,n]
#   dS[m,n]   = A[m,n] * (dAW[m,n] - s[m])                 -> bf16
#
# Two passes over the kv axis: the first accumulates the row sum, the second
# emits the output.  V is tiny and stays resident in cache, so the second pass
# only re-touches A / mask.
# ---------------------------------------------------------------------------
@triton.jit
def _ds_kernel(
    DO, A, MSK, V, DS,
    sq, skv,
    s_do_b, s_do_m, s_do_h,
    s_a_b, s_a_h, s_a_m,
    s_v_b, s_v_h, s_v_n,
    one_minus_p,
    H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
    APPLY: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    hk = h // G

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mm = offs_m < sq

    do = tl.load(
        DO + b * s_do_b + h * s_do_h + offs_m[:, None] * s_do_m + offs_d[None, :],
        mask=mm[:, None], other=0.0,
    )

    a_base = A + b * s_a_b + h * s_a_h + offs_m[:, None] * s_a_m
    k_base = MSK + b * s_a_b + h * s_a_h + offs_m[:, None] * s_a_m
    v_base = V + b * s_v_b + hk * s_v_h

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start in range(0, skv, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        nm = offs_n < skv
        v = tl.load(v_base + offs_n[:, None] * s_v_n + offs_d[None, :],
                    mask=nm[:, None], other=0.0)
        dp = tl.dot(do, tl.trans(v))
        both = mm[:, None] & nm[None, :]
        a = tl.load(a_base + offs_n[None, :], mask=both, other=0.0).to(tl.float32)
        if APPLY:
            kp = tl.load(k_base + offs_n[None, :], mask=both, other=0)
            dp = tl.where(kp != 0, dp, 0.0) / one_minus_p
        acc += tl.sum(dp * a, axis=1)

    for start in range(0, skv, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        nm = offs_n < skv
        v = tl.load(v_base + offs_n[:, None] * s_v_n + offs_d[None, :],
                    mask=nm[:, None], other=0.0)
        dp = tl.dot(do, tl.trans(v))
        both = mm[:, None] & nm[None, :]
        a = tl.load(a_base + offs_n[None, :], mask=both, other=0.0).to(tl.float32)
        if APPLY:
            kp = tl.load(k_base + offs_n[None, :], mask=both, other=0)
            dp = tl.where(kp != 0, dp, 0.0) / one_minus_p
        ds = a * (dp - acc[:, None])
        tl.store(DS + b * s_a_b + h * s_a_h + offs_m[:, None] * s_a_m + offs_n[None, :],
                 ds.to(tl.bfloat16), mask=both)


# ---------------------------------------------------------------------------
# Kernel 2: grad_value_states
#
#   dV[b,hk,j,d] = sum_{g<G} sum_m AWD[b, hk*G+g, m, j] * dO[b, m, hk*G+g, d]
# ---------------------------------------------------------------------------
@triton.jit
def _dv_kernel(
    AWD, DO, OUT,
    sq, skv,
    s_a_b, s_a_h, s_a_m,
    s_do_b, s_do_m, s_do_h,
    s_o_b, s_o_h, s_o_n,
    n_mblk, n_chunks,
    HK: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
    SPLIT: tl.constexpr, ATOMIC: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_j = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_k = tl.program_id(2)
    b = pid_bh // HK
    hk = pid_bh % HK

    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    offs_d = tl.arange(0, D)
    jm = offs_j < skv

    acc = tl.zeros([BLOCK_J, D], dtype=tl.float32)

    for c in range(pid_k, n_chunks, SPLIT):
        g = c // n_mblk
        mb = c % n_mblk
        h = hk * G + g
        offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
        mm = offs_m < sq
        aw = tl.load(
            AWD + b * s_a_b + h * s_a_h + offs_m[:, None] * s_a_m + offs_j[None, :],
            mask=mm[:, None] & jm[None, :], other=0.0,
        )
        do = tl.load(
            DO + b * s_do_b + h * s_do_h + offs_m[:, None] * s_do_m + offs_d[None, :],
            mask=mm[:, None], other=0.0,
        )
        acc += tl.dot(tl.trans(aw), do)

    optr = OUT + b * s_o_b + hk * s_o_h + offs_j[:, None] * s_o_n + offs_d[None, :]
    if ATOMIC:
        tl.atomic_add(optr, acc, mask=jm[:, None], sem="relaxed")
    else:
        tl.store(optr, acc.to(tl.bfloat16), mask=jm[:, None])


@triton.jit
def _cast_kernel(SRC, DST, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(DST + offs, tl.load(SRC + offs, mask=m, other=0.0).to(tl.bfloat16), mask=m)


def _pick_bm(sq, n_bh):
    # keep enough workgroups to fill 256 CUs
    for bm in (128, 64, 32, 16):
        if triton.cdiv(sq, bm) * n_bh >= 2048 or bm == 16:
            return bm
    return 32


@torch.no_grad()
def run(
    grad_attn_output: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_weights_dropped: torch.Tensor,
    value_states: torch.Tensor,
    dropout_mask: torch.Tensor,
    attention_dropout: float,
):
    H = 80
    HK = 8
    G = H // HK

    B = grad_attn_output.shape[0]
    SQ = grad_attn_output.shape[1]
    SKV = value_states.shape[2]
    D = value_states.shape[3]

    if isinstance(attention_dropout, torch.Tensor):
        p = float(attention_dropout.item())
    else:
        p = float(attention_dropout)
    apply_drop = p > 0.0
    one_minus_p = 1.0 - p

    dev = grad_attn_output.device

    grad_attn_scores = torch.empty(
        (B, H, SQ, SKV), dtype=torch.bfloat16, device=dev
    )

    msk = dropout_mask.view(torch.uint8) if dropout_mask.dtype == torch.bool else dropout_mask

    BM = _pick_bm(SQ, B * H)
    BN = 128 if SKV >= 128 else 64
    if BM * BN > 8192:
        BN = 64
    nw = 8 if BM * BN >= 8192 else 4

    _ds_kernel[(triton.cdiv(SQ, BM), B * H)](
        grad_attn_output, attn_weights, msk, value_states, grad_attn_scores,
        SQ, SKV,
        grad_attn_output.stride(0), grad_attn_output.stride(1), grad_attn_output.stride(2),
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2),
        value_states.stride(0), value_states.stride(1), value_states.stride(2),
        one_minus_p,
        H=H, G=G, D=D, APPLY=apply_drop,
        BLOCK_M=BM, BLOCK_N=BN,
        num_warps=nw, num_stages=2,
    )

    # ---- dV ----
    BJ = 64
    BMv = 64
    n_mblk = triton.cdiv(SQ, BMv)
    n_chunks = G * n_mblk
    base_wgs = triton.cdiv(SKV, BJ) * B * HK
    split = 1
    while split * 2 <= n_chunks and base_wgs * split < 1024:
        split *= 2

    if split == 1:
        grad_value_states = torch.empty((B, HK, SKV, D), dtype=torch.bfloat16, device=dev)
        out = grad_value_states
        atomic = False
    else:
        out = torch.zeros((B, HK, SKV, D), dtype=torch.float32, device=dev)
        atomic = True

    _dv_kernel[(triton.cdiv(SKV, BJ), B * HK, split)](
        attn_weights_dropped, grad_attn_output, out,
        SQ, SKV,
        attn_weights_dropped.stride(0), attn_weights_dropped.stride(1), attn_weights_dropped.stride(2),
        grad_attn_output.stride(0), grad_attn_output.stride(1), grad_attn_output.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        n_mblk, n_chunks,
        HK=HK, G=G, D=D, SPLIT=split, ATOMIC=atomic,
        BLOCK_J=BJ, BLOCK_M=BMv,
        num_warps=4, num_stages=2,
    )

    if atomic:
        grad_value_states = torch.empty((B, HK, SKV, D), dtype=torch.bfloat16, device=dev)
        n = out.numel()
        _cast_kernel[(triton.cdiv(n, 4096),)](out, grad_value_states, n, BLOCK=4096, num_warps=4)

    return grad_attn_scores, grad_value_states
