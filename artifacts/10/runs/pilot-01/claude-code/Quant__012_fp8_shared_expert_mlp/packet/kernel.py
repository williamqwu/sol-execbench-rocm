import torch
import triton
import triton.language as tl

FP8_MAX = tl.constexpr(448.0)
FP8_MIN = tl.constexpr(-448.0)
EPS = tl.constexpr(1e-12)
FP8 = tl.float8e4nv


# ---------------------------------------------------------------------------
# Activation quantisation: 1x128 blocks along K.
#   scale[m, kb] = clamp(amax(|x[m, kb*128:(kb+1)*128]|) / 448, min=1e-12)
#   q[m, k]      = clamp(x / scale, -448, 448) -> fp8e4m3
# Scales are stored transposed, (K//128, M), so a GEMM K-step reads a
# contiguous run of scales.
# ---------------------------------------------------------------------------
@triton.jit
def _quant_act(X, Q, S, M, K, stride_xm, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * 128 + tl.arange(0, 128)
    m_ok = rm < M
    mask = m_ok[:, None]

    x = tl.load(X + rm[:, None] * stride_xm + rk[None, :], mask=mask, other=0.0)
    x = x.to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    scale = tl.maximum(amax / FP8_MAX, EPS)
    q = x / scale[:, None]
    q = tl.minimum(tl.maximum(q, FP8_MIN), FP8_MAX)
    tl.store(Q + rm[:, None] * K + rk[None, :], q.to(FP8), mask=mask)
    tl.store(S + pid_k * M + rm, scale, mask=m_ok)


# ---------------------------------------------------------------------------
# Weight quantisation: 128x128 blocks.  The reference blocks W.T (K, N) into
# 128x128 tiles, which is exactly a 128(N) x 128(K) tile of W (N, K), and the
# scale it feeds the GEMM is indexed [nb, kb].
# ---------------------------------------------------------------------------
@triton.jit
def _quant_w(W, Q, S, K: tl.constexpr, NKB: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    rn = pid_n * 128 + tl.arange(0, 128)
    rk = pid_k * 128 + tl.arange(0, 128)
    off = rn[:, None] * K + rk[None, :]

    w = tl.load(W + off).to(tl.float32)
    amax = tl.max(tl.abs(w))
    scale = tl.maximum(amax / FP8_MAX, EPS)
    q = w / scale
    q = tl.minimum(tl.maximum(q, FP8_MIN), FP8_MAX)
    tl.store(Q + off, q.to(FP8))
    tl.store(S + pid_n * NKB + pid_k, scale)


# ---------------------------------------------------------------------------
# Fused gate+up blockwise-scaled GEMM -> SwiGLU -> requantise the intermediate.
# A program owns a (BLOCK_M, 128) output tile.  128 columns is exactly one
# activation scale block along the intermediate axis, so the requantisation of
# the SwiGLU result is entirely local -- no second pass over the intermediate.
# ---------------------------------------------------------------------------
@triton.jit
def _gemm_swiglu(
    XQ, XS, WG, WU, WGS, WUS, IQ, ISC,
    M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n: tl.constexpr = N // 128
    num_in_group = GROUP_M * num_pid_n
    group_id = pid // num_in_group
    first_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % num_in_group) % group_size_m)
    pid_n = (pid % num_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * 128 + tl.arange(0, 128)
    rm_c = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rk = tl.arange(0, 128)

    xp = XQ + rm_c[:, None] * K + rk[None, :]
    wgp = WG + rn[None, :] * K + rk[:, None]
    wup = WU + rn[None, :] * K + rk[:, None]

    acc_g = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, 128), dtype=tl.float32)

    NKB: tl.constexpr = K // 128
    for kb in range(0, NKB):
        a = tl.load(xp)
        bg = tl.load(wgp)
        bu = tl.load(wup)
        sa = tl.load(XS + kb * M + rm_c)
        sg = tl.load(WGS + pid_n * NKB + kb)
        su = tl.load(WUS + pid_n * NKB + kb)
        acc_g += tl.dot(a, bg) * (sa * sg)[:, None]
        acc_u += tl.dot(a, bu) * (sa * su)[:, None]
        xp += 128
        wgp += 128
        wup += 128

    # Reproduce the reference's intermediate rounding exactly: each GEMM output
    # is materialised as bf16, silu is an elementwise bf16 op (float math,
    # bf16 result), and the product is stored bf16 before requantisation.
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    silu = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    inter = (silu * u).to(tl.bfloat16).to(tl.float32)

    amax = tl.max(tl.abs(inter), axis=1)
    scale = tl.maximum(amax / FP8_MAX, EPS)
    q = inter / scale[:, None]
    q = tl.minimum(tl.maximum(q, FP8_MIN), FP8_MAX)

    m_ok = rm < M
    tl.store(IQ + rm[:, None] * N + rn[None, :], q.to(FP8), mask=m_ok[:, None])
    tl.store(ISC + pid_n * M + rm, scale, mask=m_ok)


# ---------------------------------------------------------------------------
# Down projection: blockwise-scaled fp8 GEMM -> bf16.
# ---------------------------------------------------------------------------
@triton.jit
def _gemm_down(
    AQ, ASC, W, WS, OUT,
    M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n: tl.constexpr = N // BLOCK_N
    num_in_group = GROUP_M * num_pid_n
    group_id = pid // num_in_group
    first_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % num_in_group) % group_size_m)
    pid_n = (pid % num_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm_c = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rk = tl.arange(0, 128)

    ap = AQ + rm_c[:, None] * K + rk[None, :]
    bp = W + rn[None, :] * K + rk[:, None]

    NKB: tl.constexpr = K // 128
    NSB: tl.constexpr = BLOCK_N // 128

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, NKB):
        a = tl.load(ap)
        b = tl.load(bp)
        sa = tl.load(ASC + kb * M + rm_c)
        d = tl.dot(a, b)
        if NSB == 1:
            sb = tl.load(WS + pid_n * NKB + kb)
            acc += d * (sa * sb)[:, None]
        else:
            sbi = pid_n * NSB + tl.arange(0, NSB)
            sb = tl.load(WS + sbi * NKB + kb)
            sbf = tl.reshape(tl.broadcast_to(sb[:, None], (NSB, 128)), (BLOCK_N,))
            acc += d * (sa[:, None] * sbf[None, :])
        ap += 128
        bp += 128

    tl.store(
        OUT + rm[:, None] * N + rn[None, :],
        acc.to(tl.bfloat16),
        mask=(rm < M)[:, None],
    )


_W_CFG = dict(num_warps=4, num_stages=1)


def _quantize_weight(w):
    N, K = w.shape
    q = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=w.device)
    s = torch.empty((N // 128, K // 128), dtype=torch.float32, device=w.device)
    _quant_w[(N // 128, K // 128)](w, q, s, K, K // 128, **_W_CFG)
    return q, s


def _cfg1(M):
    if M <= 384:
        return dict(BLOCK_M=32, GROUP_M=8), 4, 2
    if M <= 1024:
        return dict(BLOCK_M=64, GROUP_M=8), 4, 2
    return dict(BLOCK_M=128, GROUP_M=8), 8, 2


def _cfg2(M):
    if M <= 384:
        return dict(BLOCK_M=32, BLOCK_N=128, GROUP_M=16), 4, 2
    if M <= 1024:
        return dict(BLOCK_M=64, BLOCK_N=128, GROUP_M=16), 4, 2
    return dict(BLOCK_M=128, BLOCK_N=128, GROUP_M=16), 8, 2


@torch.no_grad()
def run(hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight):
    M, K = hidden_states.shape
    I = gate_proj_weight.shape[0]
    H = down_proj_weight.shape[0]
    dev = hidden_states.device

    # --- quantise activations (1x128) and all three weights (128x128) -------
    xq = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=dev)
    xs = torch.empty((K // 128, M), dtype=torch.float32, device=dev)
    BMQ = 64
    _quant_act[(triton.cdiv(M, BMQ), K // 128)](
        hidden_states, xq, xs, M, K, hidden_states.stride(0),
        BLOCK_M=BMQ, num_warps=4,
    )

    gq, gs = _quantize_weight(gate_proj_weight)
    uq, us = _quantize_weight(up_proj_weight)
    dq, ds = _quantize_weight(down_proj_weight)

    # --- fused gate/up GEMM + SwiGLU + requantise ---------------------------
    iq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    isc = torch.empty((I // 128, M), dtype=torch.float32, device=dev)

    c1, nw1, ns1 = _cfg1(M)
    grid1 = (triton.cdiv(M, c1["BLOCK_M"]) * (I // 128),)
    _gemm_swiglu[grid1](
        xq, xs, gq, uq, gs, us, iq, isc,
        M, I, K, num_warps=nw1, num_stages=ns1, **c1,
    )

    # --- down projection ----------------------------------------------------
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    c2, nw2, ns2 = _cfg2(M)
    grid2 = (triton.cdiv(M, c2["BLOCK_M"]) * (H // c2["BLOCK_N"]),)
    _gemm_down[grid2](
        iq, isc, dq, ds, out, M, H, I, num_warps=nw2, num_stages=ns2, **c2,
    )
    return out
