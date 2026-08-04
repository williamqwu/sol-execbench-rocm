import torch

import triton
import triton.language as tl

FP8_MAX = tl.constexpr(448.0)
FP8 = tl.constexpr(tl.float8e4nv)


# ---------------------------------------------------------------- quantization

@triton.jit
def _quant_act_1x128(X, Q, S, M, stride_xm, stride_qm, stride_sm,
                     BLOCK_M: tl.constexpr):
    """[M, K] bf16 -> fp8, one scale per 1x128 row-block."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * 128 + tl.arange(0, 128)
    m_ok = rm < M
    x = tl.load(X + rm[:, None] * stride_xm + rk[None, :],
                mask=m_ok[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    scale = tl.maximum(amax * (1.0 / FP8_MAX), 1e-12)
    q = x / scale[:, None]
    q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)
    tl.store(Q + rm[:, None] * stride_qm + rk[None, :], q.to(FP8),
             mask=m_ok[:, None])
    tl.store(S + rm * stride_sm + pid_k, scale, mask=m_ok)


@triton.jit
def _quant_w2(W1, Q1, S1, W2, Q2, S2,
              s1n, q1n, ss1n, s2n, q2n, ss2n,
              KB1: tl.constexpr, KB2: tl.constexpr, NBLK1: tl.constexpr):
    """Both weights, one launch: [N, K] bf16 -> fp8 per 128x128 block."""
    pid = tl.program_id(0)
    rn = tl.arange(0, 128)
    rk = tl.arange(0, 128)
    if pid < NBLK1:
        pn = pid // KB1
        pk = pid % KB1
        w = tl.load(W1 + (pn * 128 + rn)[:, None] * s1n +
                    (pk * 128 + rk)[None, :]).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(w)) * (1.0 / FP8_MAX), 1e-12)
        q = tl.minimum(tl.maximum(w * (1.0 / scale), -FP8_MAX), FP8_MAX)
        tl.store(Q1 + (pn * 128 + rn)[:, None] * q1n + (pk * 128 + rk)[None, :],
                 q.to(FP8))
        tl.store(S1 + pn * ss1n + pk, scale)
    else:
        p = pid - NBLK1
        pn = p // KB2
        pk = p % KB2
        w = tl.load(W2 + (pn * 128 + rn)[:, None] * s2n +
                    (pk * 128 + rk)[None, :]).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(w)) * (1.0 / FP8_MAX), 1e-12)
        q = tl.minimum(tl.maximum(w * (1.0 / scale), -FP8_MAX), FP8_MAX)
        tl.store(Q2 + (pn * 128 + rn)[:, None] * q2n + (pk * 128 + rk)[None, :],
                 q.to(FP8))
        tl.store(S2 + pn * ss2n + pk, scale)


# ------------------------------------------------- gemm1 + silu + quantization

@triton.jit
def _gemm1_silu_quant(A, SA, W, SW, GQ, GS,
                      M, KB: tl.constexpr, NHB: tl.constexpr,
                      stride_am, stride_sam, stride_wn, stride_swn,
                      stride_gm, stride_gsm,
                      BLOCK_M: tl.constexpr, GROUP_M: tl.constexpr,
                      EVEN_M: tl.constexpr):
    """g = A@Wg.T, u = A@Wu.T, gated = silu(g)*u, then quantize per 1x128."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    width = GROUP_M * NHB
    group_id = pid // width
    group_size = min(num_pid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + ((pid % width) % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, 128)
    rk = tl.arange(0, 128)

    if EVEN_M:
        a_ptrs = A + rm[:, None] * stride_am + rk[None, :]
    else:
        a_ptrs = A + tl.where(rm < M, rm, 0)[:, None] * stride_am + rk[None, :]
    wg_ptrs = W + (pid_n * 128 + rn)[:, None] * stride_wn + rk[None, :]
    wu_ptrs = W + ((NHB + pid_n) * 128 + rn)[:, None] * stride_wn + rk[None, :]

    acc_g = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, 128), dtype=tl.float32)

    for kb in range(0, KB):
        a = tl.load(a_ptrs)
        wg = tl.load(wg_ptrs)
        wu = tl.load(wu_ptrs)
        if EVEN_M:
            sa = tl.load(SA + rm * stride_sam + kb)
        else:
            sa = tl.load(SA + rm * stride_sam + kb, mask=rm < M, other=0.0)
        sg = tl.load(SW + pid_n * stride_swn + kb)
        su = tl.load(SW + (NHB + pid_n) * stride_swn + kb)
        acc_g += tl.dot(a, tl.trans(wg)) * (sa[:, None] * sg)
        acc_u += tl.dot(a, tl.trans(wu)) * (sa[:, None] * su)
        a_ptrs += 128
        wg_ptrs += 128
        wu_ptrs += 128

    # Match the reference, which rounds each intermediate through bf16.
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    r = (s * u).to(tl.bfloat16).to(tl.float32)

    amax = tl.max(tl.abs(r), axis=1)
    scale = tl.maximum(amax * (1.0 / FP8_MAX), 1e-12)
    q = r / scale[:, None]
    q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)

    gq_ptrs = GQ + rm[:, None] * stride_gm + (pid_n * 128 + rn)[None, :]
    if EVEN_M:
        tl.store(gq_ptrs, q.to(FP8))
        tl.store(GS + rm * stride_gsm + pid_n, scale)
    else:
        tl.store(gq_ptrs, q.to(FP8), mask=(rm < M)[:, None])
        tl.store(GS + rm * stride_gsm + pid_n, scale, mask=rm < M)


# ------------------------------------------------------ gemm2 + routing weight

@triton.jit
def _gemm2(A, SA, W, SW, RW, C,
           M, KB: tl.constexpr, NB: tl.constexpr,
           stride_am, stride_sam, stride_wn, stride_swn, stride_cm,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr,
           EVEN_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = NB * (128 // BLOCK_N)
    width = GROUP_M * num_pid_n
    group_id = pid // width
    group_size = min(num_pid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + ((pid % width) % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 128)
    swb = (pid_n * BLOCK_N) // 128

    if EVEN_M:
        a_ptrs = A + rm[:, None] * stride_am + rk[None, :]
    else:
        a_ptrs = A + tl.where(rm < M, rm, 0)[:, None] * stride_am + rk[None, :]
    w_ptrs = W + rn[:, None] * stride_wn + rk[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, KB):
        a = tl.load(a_ptrs)
        w = tl.load(w_ptrs)
        if EVEN_M:
            sa = tl.load(SA + rm * stride_sam + kb)
        else:
            sa = tl.load(SA + rm * stride_sam + kb, mask=rm < M, other=0.0)
        sw = tl.load(SW + swb * stride_swn + kb)
        acc += tl.dot(a, tl.trans(w)) * (sa[:, None] * sw)
        a_ptrs += 128
        w_ptrs += 128

    o = acc.to(tl.bfloat16).to(tl.float32)
    if EVEN_M:
        rw = tl.load(RW + rm).to(tl.float32)
    else:
        rw = tl.load(RW + rm, mask=rm < M, other=0.0).to(tl.float32)
    o = o * rw[:, None]
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :]
    if EVEN_M:
        tl.store(c_ptrs, o.to(tl.bfloat16))
    else:
        tl.store(c_ptrs, o.to(tl.bfloat16), mask=(rm < M)[:, None])


# ----------------------------------------------------------------- python glue

def quantize_act(x):
    M, K = x.shape
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((M, K // 128), dtype=torch.float32, device=x.device)
    BM = 64
    _quant_act_1x128[(triton.cdiv(M, BM), K // 128)](
        x, q, s, M, x.stride(0), q.stride(0), s.stride(0), BLOCK_M=BM,
        num_warps=2, num_stages=2)
    return q, s


def quantize_w2(w1, w2):
    N1, K1 = w1.shape
    N2, K2 = w2.shape
    q1 = torch.empty((N1, K1), dtype=torch.float8_e4m3fn, device=w1.device)
    s1 = torch.empty((N1 // 128, K1 // 128), dtype=torch.float32, device=w1.device)
    q2 = torch.empty((N2, K2), dtype=torch.float8_e4m3fn, device=w2.device)
    s2 = torch.empty((N2 // 128, K2 // 128), dtype=torch.float32, device=w2.device)
    nblk1 = (N1 // 128) * (K1 // 128)
    nblk2 = (N2 // 128) * (K2 // 128)
    _quant_w2[(nblk1 + nblk2,)](
        w1, q1, s1, w2, q2, s2,
        w1.stride(0), q1.stride(0), s1.stride(0),
        w2.stride(0), q2.stride(0), s2.stride(0),
        KB1=K1 // 128, KB2=K2 // 128, NBLK1=nblk1,
        num_warps=2, num_stages=2)
    return q1, s1, q2, s2


def _cfg(M):
    """(gemm1 BM, gemm1 warps, gemm2 BM, gemm2 warps), tuned on MI350X."""
    if M <= 1024:
        return 64, 8, 64, 8
    return 128, 8, 128, 4


def moe(hidden_states, routing_weight, gate_up_weight, down_weight):
    M, K = hidden_states.shape
    NH = gate_up_weight.shape[0] // 2
    H = down_weight.shape[0]
    dev = hidden_states.device

    bm1, nw1, bm2, nw2 = _cfg(M)

    aq, asc = quantize_act(hidden_states)
    w1q, w1s, w2q, w2s = quantize_w2(gate_up_weight, down_weight)

    gq = torch.empty((M, NH), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((M, NH // 128), dtype=torch.float32, device=dev)
    _gemm1_silu_quant[(triton.cdiv(M, bm1) * (NH // 128),)](
        aq, asc, w1q, w1s, gq, gs, M, K // 128, NH // 128,
        aq.stride(0), asc.stride(0), w1q.stride(0), w1s.stride(0),
        gq.stride(0), gs.stride(0),
        BLOCK_M=bm1, GROUP_M=1, EVEN_M=(M % bm1 == 0),
        num_warps=nw1, num_stages=2)

    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    _gemm2[(triton.cdiv(M, bm2) * (H // 128),)](
        gq, gs, w2q, w2s, routing_weight, out, M, NH // 128, H // 128,
        gq.stride(0), gs.stride(0), w2q.stride(0), w2s.stride(0), out.stride(0),
        BLOCK_M=bm2, BLOCK_N=128, GROUP_M=4, EVEN_M=(M % bm2 == 0),
        num_warps=nw2, num_stages=2)
    return out


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
):
    """FP8-quantized MoE expert computation (fused Triton implementation)."""
    return moe(hidden_states, routing_weight, gate_up_weight, down_weight)
