import math

import torch
import triton
import triton.language as tl

try:
    _ERF = tl.math.erf
except AttributeError:  # pragma: no cover
    from triton.language.extra import libdevice as _libdev
    _ERF = _libdev.erf


@triton.jit
def _gelu(x):
    return x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))


# ---------------------------------------------------------------------------
# Stage 1: conv2d(1 -> C, 3x3, stride 2, pad 1) + GELU, via a rank-9 tensor-core
# dot.  The 3x3 patch is laid out as the K axis (padded to 16).
#   in : (B, 1, FIN, TIN)      contiguous NCHW
#   out: (B, FOUT, TOUT, C)    NHWC
# ---------------------------------------------------------------------------
@triton.jit
def _conv1_gelu(
    x_ptr, w_ptr, bias_ptr, o_ptr,
    FIN, TIN, FOUT, TOUT,
    C: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_by = tl.program_id(1)
    pid_n = tl.program_id(2)

    b = pid_by // FOUT
    oy = pid_by % FOUT

    ox = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    co = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_ox = ox < TOUT
    m_co = co < C

    r = tl.arange(0, 16)
    kh = r // 3
    kw = r % 3
    vr = r < 9

    bias = tl.load(bias_ptr + co, mask=m_co, other=0.0).to(tl.float32)[None, :]
    # W: [16, BLOCK_N]  (rows >= 9 are zero)
    W = tl.load(w_ptr + r[:, None] * C + co[None, :],
                mask=vr[:, None] & m_co[None, :], other=0.0)

    it = 2 * ox[:, None] + kw[None, :] - 1
    iy = 2 * oy + kh
    iyv = iy - 1
    mm = (m_ox[:, None] & vr[None, :] & (it >= 0) & (it < TIN)
          & (iyv >= 0)[None, :] & (iyv < FIN)[None, :])

    A = tl.load(x_ptr + b * FIN * TIN + iyv[None, :] * TIN + it, mask=mm, other=0.0)
    y = _gelu(tl.dot(A, W) + bias).to(tl.bfloat16)

    off = ((b * FOUT + oy) * TOUT + ox)[:, None] * C + co[None, :]
    tl.store(o_ptr + off, y, mask=m_ox[:, None] & m_co[None, :])


# ---------------------------------------------------------------------------
# Stage 2/3: conv2d(C -> C, 3x3, stride 2, pad 1) + GELU, NHWC implicit GEMM.
#   in : (B, FIN, TIN, C)
#   w  : (9, C, C)  ([kh*3+kw][ci][co])
#   out: OUT_T == 0 -> (B, FOUT, TOUT, C)
#        OUT_T == 1 -> (B, TOUT, FOUT, C)   (transposed for the projection)
# ---------------------------------------------------------------------------
@triton.jit
def _convn_gelu(
    x_ptr, w_ptr, bias_ptr, o_ptr,
    FIN, TIN, FOUT, TOUT,
    C: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, OUT_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_by = tl.program_id(1)
    pid_n = tl.program_id(2)

    b = pid_by // FOUT
    oy = pid_by % FOUT

    ox = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    co = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_ox = ox < TOUT
    m_co = co < C

    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    rk = tl.arange(0, BLOCK_K)
    xb = x_ptr + b * FIN * TIN * C

    for kh in tl.static_range(3):
        iy = 2 * oy + kh - 1
        vy = (iy >= 0) & (iy < FIN)
        for kw in tl.static_range(3):
            it = 2 * ox + kw - 1
            mrow = m_ox & vy & (it >= 0) & (it < TIN)
            arow = xb + (iy * TIN + it) * C
            wtap = w_ptr + (kh * 3 + kw) * C * C
            for k0 in tl.range(0, C, BLOCK_K):
                ci = k0 + rk
                a = tl.load(arow[:, None] + ci[None, :], mask=mrow[:, None], other=0.0)
                wv = tl.load(wtap + ci[:, None] * C + co[None, :],
                             mask=m_co[None, :], other=0.0)
                acc = tl.dot(a, wv, acc)

    acc += tl.load(bias_ptr + co, mask=m_co, other=0.0).to(tl.float32)[None, :]
    y = _gelu(acc).to(tl.bfloat16)

    if OUT_T == 0:
        off = ((b * FOUT + oy) * TOUT + ox)[:, None] * C + co[None, :]
    else:
        off = ((b * TOUT + ox) * FOUT + oy)[:, None] * C + co[None, :]
    tl.store(o_ptr + off, y, mask=m_ox[:, None] & m_co[None, :])


# ---------------------------------------------------------------------------
# Epilogue: out = bf16(matmul_result * scale) + pe[t]
# The reference casts the scaled product to bf16 before adding pe, so we do too.
# ---------------------------------------------------------------------------
@triton.jit
def _epilogue(y_ptr, pe_ptr, scale, T, NROW, D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nb: tl.constexpr = D // BLOCK
    row = pid // nb
    d = (pid % nb) * BLOCK + tl.arange(0, BLOCK)
    if row < NROW:
        t = row % T
        v = tl.load(y_ptr + row * D + d).to(tl.float32) * scale
        v = v.to(tl.bfloat16).to(tl.float32)
        p = tl.load(pe_ptr + t * D + d).to(tl.float32)
        tl.store(y_ptr + row * D + d, (v + p).to(tl.bfloat16))


def _od(n):
    return (n - 1) // 2 + 1


def _pick_bm(tout, cap):
    bm = 16
    while bm < cap and bm < tout:
        bm *= 2
    return min(bm, cap)


def _prep_w1(t):
    """(C,1,3,3) -> (16, C), rows 9..15 zero (K padded to the MFMA tile)."""
    C = t.shape[0]
    w = torch.zeros((16, C), device=t.device, dtype=t.dtype)
    w[:9] = t.reshape(C, 9).t()
    return w


def _prep_wn(t):
    """(Co,Ci,3,3) -> (9, Ci, Co)."""
    return t.permute(2, 3, 1, 0).reshape(9, t.shape[1], t.shape[0]).contiguous()


@torch.no_grad()
def run(
    input_features: torch.Tensor,
    conv2d1_weight: torch.Tensor,
    conv2d1_bias: torch.Tensor,
    conv2d2_weight: torch.Tensor,
    conv2d2_bias: torch.Tensor,
    conv2d3_weight: torch.Tensor,
    conv2d3_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    positional_embedding: torch.Tensor,
    embed_scale: float,
):
    B, _, F0, T0 = input_features.shape
    C = conv2d1_weight.shape[0]
    dev = input_features.device

    F1, T1 = _od(F0), _od(T0)
    F2, T2 = _od(F1), _od(T1)
    F3, T3 = _od(F2), _od(T2)

    x = input_features.contiguous()

    # ---- stage 1 -----------------------------------------------------------
    w1 = _prep_w1(conv2d1_weight)
    y1 = torch.empty((B, F1, T1, C), device=dev, dtype=torch.bfloat16)
    BM = _pick_bm(T1, 32)
    grid = (triton.cdiv(T1, BM), B * F1, triton.cdiv(C, 128))
    _conv1_gelu[grid](x, w1, conv2d1_bias, y1, F0, T0, F1, T1,
                      C=C, BLOCK_M=BM, BLOCK_N=128, num_warps=4, num_stages=2)

    # ---- stage 2 -----------------------------------------------------------
    w2 = _prep_wn(conv2d2_weight)
    y2 = torch.empty((B, F2, T2, C), device=dev, dtype=torch.bfloat16)
    BM = _pick_bm(T2, 128)
    nw = 8 if BM >= 128 else 4
    grid = (triton.cdiv(T2, BM), B * F2, triton.cdiv(C, 128))
    _convn_gelu[grid](y1, w2, conv2d2_bias, y2, F1, T1, F2, T2,
                      C=C, BLOCK_M=BM, BLOCK_N=128, BLOCK_K=64, OUT_T=0,
                      num_warps=nw, num_stages=2)

    # ---- stage 3 (writes transposed so the projection sees a plain matrix) --
    w3 = _prep_wn(conv2d3_weight)
    y3 = torch.empty((B, T3, F3, C), device=dev, dtype=torch.bfloat16)
    BM = _pick_bm(T3, 128)
    nw = 8 if BM >= 128 else 4
    grid = (triton.cdiv(T3, BM), B * F3, triton.cdiv(C, 128))
    _convn_gelu[grid](y2, w3, conv2d3_bias, y3, F2, T2, F3, T3,
                      C=C, BLOCK_M=BM, BLOCK_N=128, BLOCK_K=64, OUT_T=1,
                      num_warps=nw, num_stages=2)

    # ---- projection --------------------------------------------------------
    D = conv_out_weight.shape[0]
    # reference flattens as (channel, freq); y3 rows are (freq, channel)
    wo = conv_out_weight.view(D, C, F3).permute(0, 2, 1).reshape(D, F3 * C)
    out = torch.matmul(y3.view(B * T3, F3 * C), wo.t())

    # ---- scale + positional embedding --------------------------------------
    BLOCK = 256
    nrow = B * T3
    _epilogue[(nrow * (D // BLOCK),)](out, positional_embedding, float(embed_scale),
                                      T3, nrow, D=D, BLOCK=BLOCK, num_warps=4)
    return out.view(B, T3, D)
