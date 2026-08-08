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
        nb1 = pid // KB1
        kb1 = pid % KB1
        p1 = (nb1 * 128 + tl.arange(0, 128))[:, None] * s1n + \
             (kb1 * 128 + tl.arange(0, 128))[None, :]
        w1 = tl.load(W1 + p1).to(tl.float32)
        c1 = tl.maximum(tl.max(tl.abs(w1)) * RMAX, 1e-12)
        tl.store(Q1 + p1, tl.minimum(tl.maximum(w1 / c1, -EMAX), EMAX).to(FP8))
        tl.store(S1 + nb1 * s1s + kb1, c1)
    elif pid < NP1 + NP2:
        j = pid - NP1
        nb2 = j // KB2
        kb2 = j % KB2
        p2 = (nb2 * 128 + tl.arange(0, 128))[:, None] * s2n + \
             (kb2 * 128 + tl.arange(0, 128))[None, :]
        w2 = tl.load(W2 + p2).to(tl.float32)
        c2 = tl.maximum(tl.max(tl.abs(w2)) * RMAX, 1e-12)
        tl.store(Q2 + p2, tl.minimum(tl.maximum(w2 / c2, -EMAX), EMAX).to(FP8))
        tl.store(S2 + nb2 * s2s + kb2, c2)
    else:
        h = pid - NP1 - NP2
        mb = h // KBA
        kba = h % KBA
        rm = mb * BM + tl.arange(0, BM)
        rka = kba * 128 + tl.arange(0, 128)
        mm = rm[:, None] < M
        xa = tl.load(X + rm[:, None] * sxm + rka[None, :], mask=mm,
                     other=0.0).to(tl.float32)
        ca = tl.maximum(tl.max(tl.abs(xa), axis=1) * RMAX, 1e-12)
        va = tl.minimum(tl.maximum(xa / ca[:, None], -EMAX), EMAX)
        tl.store(QA + rm[:, None] * sqm + rka[None, :], va.to(FP8), mask=mm)
        tl.store(SA + rm * sam + kba, ca, mask=rm < M)


# ------------------------------------------------------------------- gemm 1
@triton.jit
def _gemm1(A, SA, W, SW, Q, S, M, I,
           sam, ssam, swn, sswn, sqm, ssm,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 128)
    NB: tl.constexpr = BLOCK_N // 128
    IB = I // 128

    a_ptr = A + rm[:, None] * sam + rk[None, :]
    g_ptr = W + rn[:, None] * swn + rk[None, :]
    u_ptr = W + (rn + I)[:, None] * swn + rk[None, :]
    mmask = rm[:, None] < M
    sn = pid_n * NB + tl.arange(0, NB)

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in tl.range(0, NUM_K):
        a = tl.load(a_ptr, mask=mmask, other=0.0)
        bg = tl.load(g_ptr)
        bu = tl.load(u_ptr)
        sa = tl.load(SA + rm * ssam + kb, mask=rm < M, other=0.0)
        sg = tl.load(SW + sn * sswn + kb)
        su = tl.load(SW + (sn + IB) * sswn + kb)
        sge = tl.reshape(tl.broadcast_to(sg[:, None], (NB, 128)), (BLOCK_N,))
        sue = tl.reshape(tl.broadcast_to(su[:, None], (NB, 128)), (BLOCK_N,))
        acc_g += tl.dot(a, tl.trans(bg)) * (sa[:, None] * sge[None, :])
        acc_u += tl.dot(a, tl.trans(bu)) * (sa[:, None] * sue[None, :])
        a_ptr += 128
        g_ptr += 128
        u_ptr += 128

    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    y = (s * u).to(tl.bfloat16).to(tl.float32)

    yb = tl.reshape(y, (BLOCK_M, NB, 128))
    sc = tl.maximum(tl.max(tl.abs(yb), axis=2) * RMAX, 1e-12)
    sce = tl.reshape(tl.broadcast_to(sc[:, :, None], (BLOCK_M, NB, 128)),
                     (BLOCK_M, BLOCK_N))
    v = tl.minimum(tl.maximum(y / sce, -EMAX), EMAX)
    tl.store(Q + rm[:, None] * sqm + rn[None, :], v.to(FP8), mask=mmask)
    tl.store(S + rm[:, None] * ssm + sn[None, :], sc, mask=rm[:, None] < M)


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
def _cfg1(M):
    if M <= 1024:
        return (64, 128, 8, 3, 2)
    return (128, 128, 8, 2, 2)


def _cfg2(M):
    if M <= 1152:
        return (64, 128, 8, 2, 1)
    return (128, 128, 4, 2, 2)


_QBM = 64


def moe(hidden_states, routing_weight, gate_up_weight, down_weight,
        cfg1=None, cfg2=None):
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

    KB1, KB2 = H // 128, I // 128
    NP1 = (N2 // 128) * KB1
    NP2 = (H // 128) * KB2
    NPA = triton.cdiv(M, _QBM) * KB1
    _quant_all[(NP1 + NP2 + NPA,)](
        gate_up_weight, wq, wsc, down_weight, dq, dsc, hidden_states, aq, asc, M,
        gate_up_weight.stride(0), wsc.stride(0), down_weight.stride(0),
        dsc.stride(0), hidden_states.stride(0), aq.stride(0), asc.stride(0),
        KB1=KB1, KB2=KB2, KBA=KB1, NP1=NP1, NP2=NP2, BM=_QBM, num_warps=4)

    bm, bn, nw, ns, wpe = cfg1 or _cfg1(M)
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((M, I // 128), dtype=torch.float32, device=dev)
    _gemm1[(triton.cdiv(M, bm), triton.cdiv(I, bn))](
        aq, asc, wq, wsc, gq, gs, M, I,
        aq.stride(0), asc.stride(0), wq.stride(0), wsc.stride(0),
        gq.stride(0), gs.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, NUM_K=H // 128,
        num_warps=nw, num_stages=ns, waves_per_eu=wpe)

    bm, bn, nw, ns, wpe = cfg2 or _cfg2(M)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    _gemm2[(triton.cdiv(M, bm), triton.cdiv(H, bn))](
        gq, gs, dq, dsc, routing_weight, out, M,
        gq.stride(0), gs.stride(0), dq.stride(0), dsc.stride(0), out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, NUM_K=I // 128,
        num_warps=nw, num_stages=ns, waves_per_eu=wpe)
    return out
