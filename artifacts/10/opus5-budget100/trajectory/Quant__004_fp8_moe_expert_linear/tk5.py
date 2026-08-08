import torch
import triton
import triton.language as tl

FP8 = tl.float8e4nv
EMAX = tl.constexpr(448.0)
RMAX = tl.constexpr(1.0 / 448.0)


# --------------------------------------------- fused quant: 2 weights + act
# gate_up rows are emitted in interleaved 128-row-block order:
#   dst block 2j   <- gate block j   (src rows [128j, 128j+128))
#   dst block 2j+1 <- up   block j   (src rows [I+128j, ...))
# so gemm1's 256-wide N tile holds one gate block beside its up partner and
# needs a single accumulator instead of two.
@triton.jit
def _quant_all(W1, Q1, S1, W2, Q2, S2, X, QA, SA, M,
               s1n, s1s, s2n, s2s, sxm, sqm, sam,
               IB: tl.constexpr, KB1: tl.constexpr, KB2: tl.constexpr,
               NP1: tl.constexpr, NP2: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    if pid < NP1:
        nb1 = pid // KB1
        kb1 = pid % KB1
        half = nb1 // IB
        j = nb1 % IB
        dst = 2 * j + half
        rk1 = kb1 * 128 + tl.arange(0, 128)
        sp = (nb1 * 128 + tl.arange(0, 128))[:, None] * s1n + rk1[None, :]
        dp = (dst * 128 + tl.arange(0, 128))[:, None] * s1n + rk1[None, :]
        w1 = tl.load(W1 + sp).to(tl.float32)
        c1 = tl.maximum(tl.max(tl.abs(w1)) * RMAX, 1e-12)
        tl.store(Q1 + dp, tl.minimum(tl.maximum(w1 / c1, -EMAX), EMAX).to(FP8))
        tl.store(S1 + dst * s1s + kb1, c1)
    elif pid < NP1 + NP2:
        q = pid - NP1
        nb2 = q // KB2
        kb2 = q % KB2
        p2 = (nb2 * 128 + tl.arange(0, 128))[:, None] * s2n + \
             (kb2 * 128 + tl.arange(0, 128))[None, :]
        w2 = tl.load(W2 + p2).to(tl.float32)
        c2 = tl.maximum(tl.max(tl.abs(w2)) * RMAX, 1e-12)
        tl.store(Q2 + p2, tl.minimum(tl.maximum(w2 / c2, -EMAX), EMAX).to(FP8))
        tl.store(S2 + nb2 * s2s + kb2, c2)
    else:
        h = pid - NP1 - NP2
        mb = h // KB1
        kba = h % KB1
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
# One 256-wide N tile = [gate block j | up block j]. Split via slicing after
# the k-loop. Per k-step the weight scale is 2 scalars, so the rescale is a
# single fma per accumulator element.
@triton.jit
def _gemm1(A, SA, W, SW, Q, S, M,
           sam, ssam, swn, sswn, sqm, ssm,
           BLOCK_M: tl.constexpr, NUM_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, 128)
    rk = tl.arange(0, 128)

    a_ptr = A + rm[:, None] * sam + rk[None, :]
    g_ptr = W + (2 * pid_n * 128 + rn)[:, None] * swn + rk[None, :]
    u_ptr = W + ((2 * pid_n + 1) * 128 + rn)[:, None] * swn + rk[None, :]
    mmask = rm[:, None] < M

    acc_g = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    for kb in tl.range(0, NUM_K):
        a = tl.load(a_ptr, mask=mmask, other=0.0)
        bg = tl.load(g_ptr)
        bu = tl.load(u_ptr)
        sa = tl.load(SA + rm * ssam + kb, mask=rm < M, other=0.0)
        sg = tl.load(SW + (2 * pid_n) * sswn + kb)
        su = tl.load(SW + (2 * pid_n + 1) * sswn + kb)
        tg = (sa * sg)[:, None]
        tu = (sa * su)[:, None]
        acc_g = tl.math.fma(tl.dot(a, tl.trans(bg)), tg, acc_g)
        acc_u = tl.math.fma(tl.dot(a, tl.trans(bu)), tu, acc_u)
        a_ptr += 128
        g_ptr += 128
        u_ptr += 128

    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    y = (s * u).to(tl.bfloat16).to(tl.float32)

    sc = tl.maximum(tl.max(tl.abs(y), axis=1) * RMAX, 1e-12)
    v = tl.minimum(tl.maximum(y / sc[:, None], -EMAX), EMAX)
    rq = pid_n * 128 + rn
    tl.store(Q + rm[:, None] * sqm + rq[None, :], v.to(FP8), mask=mmask)
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
        if NB == 1:
            sb = tl.load(SW + pid_n * sswn + kb)
            acc = tl.math.fma(tl.dot(a, tl.trans(b)), (sa * sb)[:, None], acc)
        else:
            sv = tl.load(SW + (pid_n * NB + tl.arange(0, NB)) * sswn + kb)
            sbe = tl.reshape(tl.broadcast_to(sv[:, None], (NB, 128)), (BLOCK_N,))
            acc = tl.math.fma(tl.dot(a, tl.trans(b)),
                              sa[:, None] * sbe[None, :], acc)
        a_ptr += 128
        b_ptr += 128
    r = tl.load(R + rm, mask=rm < M, other=0.0).to(tl.float32)
    o = acc.to(tl.bfloat16).to(tl.float32) * r[:, None]
    tl.store(C + rm[:, None] * scm + rn[None, :], o.to(tl.bfloat16), mask=mmask)


_NUM_CU = 256


def _cfg1(M, I):
    """Pick BLOCK_M by wave quantization: both tile sizes cost the same per
    output element, so the winner is whichever wastes less of the last wave.
    A bm=64 tile does half the work of a bm=128 tile, so weight the wave count
    by bm before comparing."""
    nt = I // 128
    c64 = -(-(-(-M // 64) * nt) // _NUM_CU) * 64
    c128 = -(-(-(-M // 128) * nt) // _NUM_CU) * 128
    if c64 < c128:
        return (64, 8, 3, 2)
    return (128, 8, 2, 2)


def _cfg2(M, H):
    """bm=64/nw=8 only pays off while the grid still fits in ~2 waves;
    past that the wider bm=128/nw=4 tile amortizes the weight traffic."""
    if -(-M // 64) * (H // 128) <= 2 * _NUM_CU:
        return (64, 128, 8, 2, 2)
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
        IB=I // 128, KB1=KB1, KB2=KB2, NP1=NP1, NP2=NP2, BM=_QBM, num_warps=4)

    bm, nw, ns, wpe = cfg1 or _cfg1(M, I)
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((M, I // 128), dtype=torch.float32, device=dev)
    _gemm1[(triton.cdiv(M, bm), I // 128)](
        aq, asc, wq, wsc, gq, gs, M,
        aq.stride(0), asc.stride(0), wq.stride(0), wsc.stride(0),
        gq.stride(0), gs.stride(0),
        BLOCK_M=bm, NUM_K=H // 128, num_warps=nw, num_stages=ns,
        waves_per_eu=wpe)

    bm, bn, nw, ns, wpe = cfg2 or _cfg2(M, H)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    _gemm2[(triton.cdiv(M, bm), triton.cdiv(H, bn))](
        gq, gs, dq, dsc, routing_weight, out, M,
        gq.stride(0), gs.stride(0), dq.stride(0), dsc.stride(0), out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, NUM_K=I // 128,
        num_warps=nw, num_stages=ns, waves_per_eu=wpe)
    return out
