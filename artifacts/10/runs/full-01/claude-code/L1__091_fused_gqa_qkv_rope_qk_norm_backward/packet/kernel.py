import torch
import triton
import triton.language as tl


@triton.jit
def _rb(x):
    # Round a float32 value through bfloat16, matching torch materialising a
    # bf16 intermediate tensor.
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _fused_bwd_row(
    gq_ptr, gk_ptr, gv_ptr,
    qpre_ptr, kpre_ptr,
    cos_ptr, sin_ptr,
    qw_ptr, kw_ptr,
    qrstd_ptr, krstd_ptr,
    gqkv_ptr, gcos_ptr, gsin_ptr,
    gqwp_ptr, gkwp_ptr,
    S,
    HD: tl.constexpr,
    HALF: tl.constexpr,
    NH: tl.constexpr,
    NKV: tl.constexpr,
    QKV: tl.constexpr,
    BH: tl.constexpr,
):
    """One program per (batch, token). Head dim is split into contiguous lo/hi
    halves so the RoPE half-rotation becomes a register swap instead of a
    strided reload."""
    m = tl.program_id(0)
    b = m // S
    s = m % S

    e = tl.arange(0, HALF)

    cbase = m * HD
    cos_lo = tl.load(cos_ptr + cbase + e).to(tl.float32)
    cos_hi = tl.load(cos_ptr + cbase + HALF + e).to(tl.float32)
    sin_lo = tl.load(sin_ptr + cbase + e).to(tl.float32)
    sin_hi = tl.load(sin_ptr + cbase + HALF + e).to(tl.float32)

    gcos_q_lo = tl.zeros([HALF], dtype=tl.float32)
    gcos_q_hi = tl.zeros([HALF], dtype=tl.float32)
    gsin_q_lo = tl.zeros([HALF], dtype=tl.float32)
    gsin_q_hi = tl.zeros([HALF], dtype=tl.float32)
    gqw_lo = tl.zeros([HALF], dtype=tl.float32)
    gqw_hi = tl.zeros([HALF], dtype=tl.float32)

    # ---------------- query path ----------------
    w_lo = tl.load(qw_ptr + e).to(tl.float32)
    w_hi = tl.load(qw_ptr + HALF + e).to(tl.float32)

    for h0 in tl.range(0, NH, BH):
        hs = h0 + tl.arange(0, BH)
        rowbase = (b * NH + hs) * S + s              # [BH]
        base = rowbase[:, None] * HD + e[None, :]

        go_lo = tl.load(gq_ptr + base).to(tl.float32)
        go_hi = tl.load(gq_ptr + base + HALF).to(tl.float32)
        x_lo = tl.load(qpre_ptr + base).to(tl.float32)
        x_hi = tl.load(qpre_ptr + base + HALF).to(tl.float32)
        r = tl.load(qrstd_ptr + rowbase).to(tl.float32)[:, None]

        # --- rms-norm backward ---
        gqw_lo += tl.sum(go_lo * (x_lo * r), 0)
        gqw_hi += tl.sum(go_hi * (x_hi * r), 0)

        grad_rstd = (tl.sum(go_lo * w_lo[None, :] * x_lo, 1)
                     + tl.sum(go_hi * w_hi[None, :] * x_hi, 1))[:, None]
        r3 = r * r * r
        c = grad_rstd * (-r3 / HD)
        g1_lo = _rb(go_lo * w_lo[None, :] * r + c * x_lo)
        g1_hi = _rb(go_hi * w_hi[None, :] * r + c * x_hi)

        # --- rope backward ---
        # grad_rotated_inv = cat(g1_hi, -g1_lo)
        gx_lo = _rb(_rb(g1_lo * cos_lo[None, :]) + _rb(g1_hi * sin_lo[None, :]))
        gx_hi = _rb(_rb(g1_hi * cos_hi[None, :]) + _rb(-g1_lo * sin_hi[None, :]))

        # x_original: x_rotated_inv = cat(-x_hi, x_lo)
        xo_lo = _rb(_rb(x_lo * cos_lo[None, :]) + _rb(-x_hi * sin_lo[None, :]))
        xo_hi = _rb(_rb(x_hi * cos_hi[None, :]) + _rb(x_lo * sin_hi[None, :]))

        gcos_q_lo += tl.sum(_rb(g1_lo * xo_lo), 0)
        gcos_q_hi += tl.sum(_rb(g1_hi * xo_hi), 0)
        # x_original_rotated = cat(-xo_hi, xo_lo)
        gsin_q_lo += tl.sum(_rb(g1_lo * -xo_hi), 0)
        gsin_q_hi += tl.sum(_rb(g1_hi * xo_lo), 0)

        obase = m * QKV + hs[:, None] * HD + e[None, :]
        tl.store(gqkv_ptr + obase, gx_lo.to(tl.bfloat16))
        tl.store(gqkv_ptr + obase + HALF, gx_hi.to(tl.bfloat16))

    # ---------------- key path ----------------
    w_lo = tl.load(kw_ptr + e).to(tl.float32)
    w_hi = tl.load(kw_ptr + HALF + e).to(tl.float32)

    hs = tl.arange(0, NKV)
    rowbase = (b * NKV + hs) * S + s
    base = rowbase[:, None] * HD + e[None, :]

    go_lo = tl.load(gk_ptr + base).to(tl.float32)
    go_hi = tl.load(gk_ptr + base + HALF).to(tl.float32)
    x_lo = tl.load(kpre_ptr + base).to(tl.float32)
    x_hi = tl.load(kpre_ptr + base + HALF).to(tl.float32)
    r = tl.load(krstd_ptr + rowbase).to(tl.float32)[:, None]

    gkw_lo = tl.sum(go_lo * (x_lo * r), 0)
    gkw_hi = tl.sum(go_hi * (x_hi * r), 0)

    grad_rstd = (tl.sum(go_lo * w_lo[None, :] * x_lo, 1)
                 + tl.sum(go_hi * w_hi[None, :] * x_hi, 1))[:, None]
    r3 = r * r * r
    c = grad_rstd * (-r3 / HD)
    g1_lo = _rb(go_lo * w_lo[None, :] * r + c * x_lo)
    g1_hi = _rb(go_hi * w_hi[None, :] * r + c * x_hi)

    gx_lo = _rb(_rb(g1_lo * cos_lo[None, :]) + _rb(g1_hi * sin_lo[None, :]))
    gx_hi = _rb(_rb(g1_hi * cos_hi[None, :]) + _rb(-g1_lo * sin_hi[None, :]))

    xo_lo = _rb(_rb(x_lo * cos_lo[None, :]) + _rb(-x_hi * sin_lo[None, :]))
    xo_hi = _rb(_rb(x_hi * cos_hi[None, :]) + _rb(x_lo * sin_hi[None, :]))

    gcos_k_lo = tl.sum(_rb(g1_lo * xo_lo), 0)
    gcos_k_hi = tl.sum(_rb(g1_hi * xo_hi), 0)
    gsin_k_lo = tl.sum(_rb(g1_lo * -xo_hi), 0)
    gsin_k_hi = tl.sum(_rb(g1_hi * xo_lo), 0)

    obase = m * QKV + NH * HD + hs[:, None] * HD + e[None, :]
    tl.store(gqkv_ptr + obase, gx_lo.to(tl.bfloat16))
    tl.store(gqkv_ptr + obase + HALF, gx_hi.to(tl.bfloat16))

    # ---------------- value path (transpose only) ----------------
    vbase = m * QKV + (NH + NKV) * HD + hs[:, None] * HD + e[None, :]
    tl.store(gqkv_ptr + vbase, tl.load(gv_ptr + base))
    tl.store(gqkv_ptr + vbase + HALF, tl.load(gv_ptr + base + HALF))

    # ---------------- reductions ----------------
    tl.store(gcos_ptr + cbase + e,
             _rb(_rb(gcos_q_lo) + _rb(gcos_k_lo)).to(tl.bfloat16))
    tl.store(gcos_ptr + cbase + HALF + e,
             _rb(_rb(gcos_q_hi) + _rb(gcos_k_hi)).to(tl.bfloat16))
    tl.store(gsin_ptr + cbase + e,
             _rb(_rb(gsin_q_lo) + _rb(gsin_k_lo)).to(tl.bfloat16))
    tl.store(gsin_ptr + cbase + HALF + e,
             _rb(_rb(gsin_q_hi) + _rb(gsin_k_hi)).to(tl.bfloat16))
    wrow = m * 2 * HD
    tl.store(gqwp_ptr + wrow + e, gqw_lo)
    tl.store(gqwp_ptr + wrow + HALF + e, gqw_hi)
    tl.store(gqwp_ptr + wrow + HD + e, gkw_lo)
    tl.store(gqwp_ptr + wrow + HD + HALF + e, gkw_hi)


@triton.jit
def _transpose_kernel(src, dst, M, N,
                      BM: tl.constexpr, BN: tl.constexpr):
    """dst[N, M] = src[M, N].T  -- tiled so both sides stay coalesced."""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    v = tl.load(src + rm[:, None] * N + rn[None, :], mask=mask, other=0)
    tl.store(dst + rn[:, None] * M + rm[None, :], tl.trans(v), mask=mask.T)


def _transpose(g):
    M, N = g.shape
    o = torch.empty((N, M), dtype=g.dtype, device=g.device)
    BM = BN = 64
    _transpose_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        g, o, M, N, BM=BM, BN=BN, num_warps=4)
    return o


_NH, _NKV, _HD, _QKV = 32, 8, 128, 6144


@torch.no_grad()
def run(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    query_pre_norm: torch.Tensor,
    key_pre_norm: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    eps: float,
):
    bsz, seq_len, hidden_size = hidden_states.shape
    M = bsz * seq_len
    dev = hidden_states.device

    grad_query = grad_query.contiguous()
    grad_key = grad_key.contiguous()
    grad_value = grad_value.contiguous()
    query_pre_norm = query_pre_norm.contiguous()
    key_pre_norm = key_pre_norm.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    q_rstd = q_rstd.contiguous()
    k_rstd = k_rstd.contiguous()

    grad_qkv = torch.empty((M, _QKV), dtype=torch.bfloat16, device=dev)
    grad_cos = torch.empty((M, _HD), dtype=torch.bfloat16, device=dev)
    grad_sin = torch.empty((M, _HD), dtype=torch.bfloat16, device=dev)
    gwp = torch.empty((M, 2 * _HD), dtype=torch.float32, device=dev)
    gqwp = gwp[:, :_HD]
    gkwp = gwp[:, _HD:]

    _fused_bwd_row[(M,)](
        grad_query, grad_key, grad_value,
        query_pre_norm, key_pre_norm,
        cos, sin,
        q_norm_weight, k_norm_weight,
        q_rstd, k_rstd,
        grad_qkv, grad_cos, grad_sin,
        gwp, gwp,
        seq_len,
        HD=_HD, HALF=_HD // 2,
        NH=_NH, NKV=_NKV, QKV=_QKV,
        BH=(2 if M >= 4096 else 16), num_warps=1, num_stages=1,
    )

    grad_hidden_states = torch.mm(grad_qkv, qkv_weight).view(bsz, seq_len, hidden_size)
    # For tall A, an explicit tiled transpose + NN GEMM beats hipBLASLt's TN
    # kernel; below the crossover the extra launch costs more than it saves.
    hidden_flat = hidden_states.reshape(M, hidden_size)
    if M >= 1024:
        grad_qkv_weight = torch.mm(_transpose(grad_qkv), hidden_flat)
    else:
        grad_qkv_weight = torch.mm(grad_qkv.t(), hidden_flat)

    gw = gwp.sum(0)
    grad_q_norm_weight = gw[:_HD].to(torch.bfloat16)
    grad_k_norm_weight = gw[_HD:].to(torch.bfloat16)

    return (
        grad_hidden_states,
        grad_cos.view(bsz, seq_len, _HD),
        grad_sin.view(bsz, seq_len, _HD),
        grad_qkv_weight,
        grad_q_norm_weight,
        grad_k_norm_weight,
    )
