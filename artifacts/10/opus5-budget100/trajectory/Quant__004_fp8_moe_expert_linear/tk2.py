import torch
import triton
import triton.language as tl

FP8 = tl.float8e4nv
EMAX = tl.constexpr(448.0)
RMAX = tl.constexpr(1.0 / 448.0)


# ------------------------------------------------ fused quant: 2 weights + act
@triton.jit
def _quant_all(W1, Q1, S1, W2, Q2, S2, X, QA, SA, M,
               s1n, s1s, s2n, s2s, sxm, sqm, sam,
               KB1: tl.constexpr, KB2: tl.constexpr, KBA: tl.constexpr,
               NP1: tl.constexpr, NP2: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    if pid < NP1:
        nb = pid // KB1
        kb = pid % KB1
        p = (nb * 128 + tl.arange(0, 128))[:, None] * s1n + \
            (kb * 128 + tl.arange(0, 128))[None, :]
        w = tl.load(W1 + p).to(tl.float32)
        sc = tl.maximum(tl.max(tl.abs(w)) * RMAX, 1e-12)
        tl.store(Q1 + p, tl.minimum(tl.maximum(w / sc, -EMAX), EMAX).to(FP8))
        tl.store(S1 + nb * s1s + kb, sc)
    elif pid < NP1 + NP2:
        q = pid - NP1
        nb = q // KB2
        kb = q % KB2
        p = (nb * 128 + tl.arange(0, 128))[:, None] * s2n + \
            (kb * 128 + tl.arange(0, 128))[None, :]
        w = tl.load(W2 + p).to(tl.float32)
        sc = tl.maximum(tl.max(tl.abs(w)) * RMAX, 1e-12)
        tl.store(Q2 + p, tl.minimum(tl.maximum(w / sc, -EMAX), EMAX).to(FP8))
        tl.store(S2 + nb * s2s + kb, sc)
    else:
        q = pid - NP1 - NP2
        mb = q // KBA
        kb = q % KBA
        rm = mb * BM + tl.arange(0, BM)
        rk = kb * 128 + tl.arange(0, 128)
        mm = rm[:, None] < M
        x = tl.load(X + rm[:, None] * sxm + rk[None, :], mask=mm,
                    other=0.0).to(tl.float32)
        sc = tl.maximum(tl.max(tl.abs(x), axis=1) * RMAX, 1e-12)
        v = tl.minimum(tl.maximum(x / sc[:, None], -EMAX), EMAX)
        tl.store(QA + rm[:, None] * sqm + rk[None, :], v.to(FP8), mask=mm)
        tl.store(SA + rm * sam + kb, sc, mask=rm < M)


# ------------------------------------------------------------------- gemm 1
@triton.jit
def _gemm1(A, SA, W, SW, Q, S, M, I,
           sam, ssam, swn, sswn, sqm, ssm,
           BLOCK_M: tl.constexpr, NUM_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * 128 + tl.arange(0, 128)
    rk = tl.arange(0, 128)

    a_ptr = A + rm[:, None] * sam + rk[None, :]
    g_ptr = W + rn[:, None] * swn + rk[None, :]
    u_ptr = W + (rn + I)[:, None] * swn + rk[None, :]
    mmask = rm[:, None] < M

    acc_g = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    for kb in tl.range(0, NUM_K):
        a = tl.load(a_ptr, mask=mmask, other=0.0)
        bg = tl.load(g_ptr)
        bu = tl.load(u_ptr)
        sa = tl.load(SA + rm * ssam + kb, mask=rm < M, other=0.0)
        sg = tl.load(SW + pid_n * sswn + kb)
        su = tl.load(SW + (pid_n + I // 128) * sswn + kb)
        acc_g += tl.dot(a, tl.trans(bg)) * (sa[:, None] * sg)
        acc_u += tl.dot(a, tl.trans(bu)) * (sa[:, None] * su)
        a_ptr += 128
        g_ptr += 128
        u_ptr += 128

    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    y = (s * u).to(tl.bfloat16).to(tl.float32)

    sc = tl.maximum(tl.max(tl.abs(y), axis=1) * RMAX, 1e-12)
    v = tl.minimum(tl.maximum(y / sc[:, None], -EMAX), EMAX)
    tl.store(Q + rm[:, None] * sqm + rn[None, :], v.to(FP8), mask=mmask)
    tl.store(S + rm * ssm + pid_n, sc, mask=rm < M)


# ------------------------------------------------------------------- gemm 2
@triton.jit
def _gemm2(A, SA, W, SW, R, C, M,
           sam, ssam, swn, sswn, scm,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 128)
    a_ptr = A + rm[:, None] * sam + rk[None, :]
    b_ptr = W + rn[:, None] * swn + rk[None, :]
    mmask = rm[:, None] < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    NB: tl.constexpr = BLOCK_N // 128
    for kb in tl.range(0, NUM_K):
        a = tl.load(a_ptr, mask=mmask, other=0.0)
        b = tl.load(b_ptr)
        sa = tl.load(SA + rm * ssam + kb, mask=rm < M, other=0.0)
        sb = tl.load(SW + (pid_n * NB + tl.arange(0, NB)) * sswn + kb)
        sbe = tl.reshape(tl.broadcast_to(sb[:, None], (NB, 128)), (BLOCK_N,))
        acc += tl.dot(a, tl.trans(b)) * (sa[:, None] * sbe[None, :])
        a_ptr += 128
        b_ptr += 128
    r = tl.load(R + rm, mask=rm < M, other=0.0).to(tl.float32)
    o = acc.to(tl.bfloat16).to(tl.float32) * r[:, None]
    tl.store(C + rm[:, None] * scm + rn[None, :], o.to(tl.bfloat16), mask=mmask)


# ---------------------------------------------------------------- dispatch
# (BLOCK_M, num_warps, num_stages) for gemm1, keyed by M threshold
def _cfg1(M):
    if M <= 1024:
        return (64, 8, 3)
    return (128, 8, 2)


# (BLOCK_M, BLOCK_N, num_warps, num_stages) for gemm2
def _cfg2(M):
    if M <= 1152:
        return (64, 128, 8, 2)
    return (128, 128, 4, 2)


_QBM = 64


def moe(hidden_states, routing_weight, gate_up_weight, down_weight):
    M, H = hidden_states.shape
    N2 = gate_up_weight.shape[0]
    I = N2 // 2
    dev = hidden_states.device

    aq = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=dev)
    asc = torch.empty((M, H // 128), dtype=torch.float32, device=dev)
    wq = torch.empty((N2, H), dtype=torch.float8_e4m3fn, device=dev)
    wsc = torch.empty((N2 // 128, H // 128), dtype=torch.float32, device=dev)
    dq = torch.empty((H, I), dtype=torch.float8_e4m3fn, device=dev)
    dsc = torch.empty((H // 128, I // 128), dtype=torch.float32, device=dev)

    KB1, KB2, KBA = H // 128, I // 128, H // 128
    NP1 = (N2 // 128) * KB1
    NP2 = (H // 128) * KB2
    NPA = triton.cdiv(M, _QBM) * KBA
    _quant_all[(NP1 + NP2 + NPA,)](
        gate_up_weight, wq, wsc, down_weight, dq, dsc, hidden_states, aq, asc, M,
        gate_up_weight.stride(0), wsc.stride(0), down_weight.stride(0),
        dsc.stride(0), hidden_states.stride(0), aq.stride(0), asc.stride(0),
        KB1=KB1, KB2=KB2, KBA=KBA, NP1=NP1, NP2=NP2, BM=_QBM, num_warps=4)

    bm, nw, ns = _cfg1(M)
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((M, I // 128), dtype=torch.float32, device=dev)
    _gemm1[(triton.cdiv(M, bm), I // 128)](
        aq, asc, wq, wsc, gq, gs, M, I,
        aq.stride(0), asc.stride(0), wq.stride(0), wsc.stride(0),
        gq.stride(0), gs.stride(0),
        BLOCK_M=bm, NUM_K=H // 128, num_warps=nw, num_stages=ns)

    bm, bn, nw, ns = _cfg2(M)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    _gemm2[(triton.cdiv(M, bm), triton.cdiv(H, bn))](
        gq, gs, dq, dsc, routing_weight, out, M,
        gq.stride(0), gs.stride(0), dq.stride(0), dsc.stride(0), out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, NUM_K=I // 128, num_warps=nw, num_stages=ns)
    return out
