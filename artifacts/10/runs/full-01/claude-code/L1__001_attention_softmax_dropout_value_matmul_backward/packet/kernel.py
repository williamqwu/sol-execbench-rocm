import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A: grad_attn_scores
#
#   dP_ij      = (dO_i . V_j)                     [f32 accum over head_dim]
#   g_ij       = dP_ij * mask_ij / (1 - p)
#   s_i        = sum_j g_ij * W_ij
#   dS_ij      = W_ij * (g_ij - s_i)              -> bf16
#
# Two passes over the kv axis; dP is recomputed in the second pass rather than
# materialised (recompute is ~5x cheaper than the extra HBM round trip here).
# ---------------------------------------------------------------------------
@triton.jit
def _ds_kernel(
    DO, W, MSK, V, DS,
    SQ, SKV,
    s_do_b, s_do_m, s_do_h,
    s_w_b, s_w_h, s_w_m,
    s_v_b, s_v_h, s_v_n,
    scale,
    H: tl.constexpr, NG: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    hk = h // NG

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < SQ

    do = tl.load(
        DO + b * s_do_b + offs_m[:, None] * s_do_m + h * s_do_h + offs_d[None, :],
        mask=m_mask[:, None], other=0.0,
    )

    w_base = W + b * s_w_b + h * s_w_h + offs_m[:, None] * s_w_m
    k_base = MSK + b * s_w_b + h * s_w_h + offs_m[:, None] * s_w_m
    v_base = V + b * s_v_b + hk * s_v_h
    d_base = DS + b * s_w_b + h * s_w_h + offs_m[:, None] * s_w_m

    # ---- pass 1: row sums -------------------------------------------------
    s = tl.zeros([BM], tl.float32)
    for start in range(0, SKV, BN):
        offs_n = start + tl.arange(0, BN)
        n_mask = offs_n < SKV
        full = m_mask[:, None] & n_mask[None, :]
        v = tl.load(
            v_base + offs_n[:, None] * s_v_n + offs_d[None, :],
            mask=n_mask[:, None], other=0.0,
        )
        dp = tl.dot(do, tl.trans(v))
        w = tl.load(w_base + offs_n[None, :], mask=full, other=0.0).to(tl.float32)
        mk = tl.load(k_base + offs_n[None, :], mask=full, other=0).to(tl.float32)
        s += tl.sum(dp * mk * w, 1)
    s = s * scale

    # ---- pass 2: gradient -------------------------------------------------
    for start in range(0, SKV, BN):
        offs_n = start + tl.arange(0, BN)
        n_mask = offs_n < SKV
        full = m_mask[:, None] & n_mask[None, :]
        v = tl.load(
            v_base + offs_n[:, None] * s_v_n + offs_d[None, :],
            mask=n_mask[:, None], other=0.0,
        )
        dp = tl.dot(do, tl.trans(v))
        w = tl.load(w_base + offs_n[None, :], mask=full, other=0.0).to(tl.float32)
        mk = tl.load(k_base + offs_n[None, :], mask=full, other=0).to(tl.float32)
        g = dp * mk * scale
        ds = w * (g - s[:, None])
        tl.store(d_base + offs_n[None, :], ds.to(tl.bfloat16), mask=full)


# ---------------------------------------------------------------------------
# Kernel B: grad_value_states
#
#   dV[b,hk,j,:] = sum_{g<NG} sum_i Wd[b, hk*NG+g, i, j] * dO[b, i, hk*NG+g, :]
# ---------------------------------------------------------------------------
@triton.jit
def _dv_kernel(
    WD, DO, DV,
    SQ, SKV,
    s_do_b, s_do_m, s_do_h,
    s_w_b, s_w_h, s_w_m,
    s_dv_b, s_dv_h, s_dv_n,
    HKV: tl.constexpr, NG: tl.constexpr,
    GPS: tl.constexpr, NSPLIT: tl.constexpr, ATOMIC: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    sp = pid % NSPLIT
    tmp = pid // NSPLIT
    hk = tmp % HKV
    b = tmp // HKV

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    n_mask = offs_n < SKV

    acc = tl.zeros([BN, D], tl.float32)
    for gi in tl.static_range(GPS):
        h = hk * NG + sp * GPS + gi
        wd_base = WD + b * s_w_b + h * s_w_h
        do_base = DO + b * s_do_b + h * s_do_h
        for start in range(0, SQ, BM):
            offs_m = start + tl.arange(0, BM)
            m_mask = offs_m < SQ
            wd = tl.load(
                wd_base + offs_m[:, None] * s_w_m + offs_n[None, :],
                mask=m_mask[:, None] & n_mask[None, :], other=0.0,
            )
            do = tl.load(
                do_base + offs_m[:, None] * s_do_m + offs_d[None, :],
                mask=m_mask[:, None], other=0.0,
            )
            acc += tl.dot(tl.trans(wd), do)

    ptrs = DV + b * s_dv_b + hk * s_dv_h + offs_n[:, None] * s_dv_n + offs_d[None, :]
    if ATOMIC:
        tl.atomic_add(ptrs, acc, mask=n_mask[:, None])
    else:
        tl.store(ptrs, acc.to(tl.bfloat16), mask=n_mask[:, None])


def _pick_split(nblocks_base, ng):
    for ns in (1, 2, 5, 10):
        if ns > ng:
            break
        if ng % ns:
            continue
        if nblocks_base * ns >= 1024:
            return ns
    return ng if nblocks_base * ng < 1024 else 1


@torch.no_grad()
def run(
    grad_attn_output: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_weights_dropped: torch.Tensor,
    value_states: torch.Tensor,
    dropout_mask: torch.Tensor,
    attention_dropout: float,
):
    B, SQ, H, D = grad_attn_output.shape
    HKV = value_states.shape[1]
    SKV = value_states.shape[2]
    NG = H // HKV

    grad_attn_output = grad_attn_output.contiguous()
    attn_weights = attn_weights.contiguous()
    attn_weights_dropped = attn_weights_dropped.contiguous()
    value_states = value_states.contiguous()
    dropout_mask = dropout_mask.contiguous()

    scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    ds = torch.empty((B, H, SQ, SKV), dtype=torch.bfloat16,
                     device=grad_attn_output.device)

    s_do_b, s_do_m, s_do_h = (grad_attn_output.stride(0),
                              grad_attn_output.stride(1),
                              grad_attn_output.stride(2))
    s_w_b, s_w_h, s_w_m = (attn_weights.stride(0), attn_weights.stride(1),
                           attn_weights.stride(2))
    s_v_b, s_v_h, s_v_n = (value_states.stride(0), value_states.stride(1),
                           value_states.stride(2))

    BM = 128 if SQ >= 384 else 64
    BN = 64
    grid_a = (B * H, triton.cdiv(SQ, BM))
    _ds_kernel[grid_a](
        grad_attn_output, attn_weights, dropout_mask, value_states, ds,
        SQ, SKV,
        s_do_b, s_do_m, s_do_h,
        s_w_b, s_w_h, s_w_m,
        s_v_b, s_v_h, s_v_n,
        scale,
        H=H, NG=NG, BM=BM, BN=BN, D=D,
        num_warps=8, num_stages=2,
    )

    # ---- dV ---------------------------------------------------------------
    BMv = 64
    BNv = 64
    nb_base = B * HKV * triton.cdiv(SKV, BNv)
    nsplit = _pick_split(nb_base, NG)
    if attention_dropout <= 0.0:
        pass
    if nsplit > 1:
        dv_buf = torch.zeros((B, HKV, SKV, D), dtype=torch.float32,
                             device=grad_attn_output.device)
        atomic = True
    else:
        dv_buf = torch.empty((B, HKV, SKV, D), dtype=torch.bfloat16,
                             device=grad_attn_output.device)
        atomic = False

    grid_b = (B * HKV * nsplit, triton.cdiv(SKV, BNv))
    _dv_kernel[grid_b](
        attn_weights_dropped, grad_attn_output, dv_buf,
        SQ, SKV,
        s_do_b, s_do_m, s_do_h,
        s_w_b, s_w_h, s_w_m,
        dv_buf.stride(0), dv_buf.stride(1), dv_buf.stride(2),
        HKV=HKV, NG=NG, GPS=NG // nsplit, NSPLIT=nsplit, ATOMIC=atomic,
        BM=BMv, BN=BNv, D=D,
        num_warps=4, num_stages=2,
    )

    dv = dv_buf.to(torch.bfloat16) if atomic else dv_buf
    return ds, dv
