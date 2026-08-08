import sys, torch, time, triton, triton.language as tl, itertools
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import tk3 as tk

dev = 'cuda'
FP8 = tl.float8e4nv
EMAX = tl.constexpr(448.0)
RMAX = tl.constexpr(1.0 / 448.0)


def bench(fn, n=60, w=25):
    try:
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
    except Exception as e:
        return 1e9
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


# no-scale baseline: how fast can this loop go at all?
@triton.jit
def _g1_noscale(A, W, Q, M, I, sam, swn, sqm,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_K: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 128)
    ap = A + rm[:, None] * sam + rk[None, :]
    gp = W + rn[:, None] * swn + rk[None, :]
    up = W + (rn + I)[:, None] * swn + rk[None, :]
    mm = rm[:, None] < M
    ag = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    au = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in tl.range(0, NUM_K):
        a = tl.load(ap, mask=mm, other=0.0)
        ag = tl.dot(a, tl.trans(tl.load(gp)), ag)
        au = tl.dot(a, tl.trans(tl.load(up)), au)
        ap += 128
        gp += 128
        up += 128
    y = ag + au
    tl.store(Q + rm[:, None] * sqm + rn[None, :], y.to(FP8), mask=mm)


# BLOCK_K = 256: two scale groups per loop iteration
@triton.jit
def _g1_k256(A, SA, W, SW, Q, S, M, I, sam, ssam, swn, sswn, sqm, ssm,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_K: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 128)
    NB: tl.constexpr = BLOCK_N // 128
    IB = I // 128
    ap = A + rm[:, None] * sam + rk[None, :]
    gp = W + rn[:, None] * swn + rk[None, :]
    up = W + (rn + I)[:, None] * swn + rk[None, :]
    mm = rm[:, None] < M
    sn = pn * NB + tl.arange(0, NB)
    ag = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    au = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in tl.range(0, NUM_K // 2):
        a0 = tl.load(ap, mask=mm, other=0.0)
        a1 = tl.load(ap + 128, mask=mm, other=0.0)
        g0 = tl.load(gp)
        g1 = tl.load(gp + 128)
        u0 = tl.load(up)
        u1 = tl.load(up + 128)
        sa0 = tl.load(SA + rm * ssam + 2 * kb, mask=rm < M, other=0.0)
        sa1 = tl.load(SA + rm * ssam + 2 * kb + 1, mask=rm < M, other=0.0)
        sg0 = tl.load(SW + sn * sswn + 2 * kb)
        sg1 = tl.load(SW + sn * sswn + 2 * kb + 1)
        su0 = tl.load(SW + (sn + IB) * sswn + 2 * kb)
        su1 = tl.load(SW + (sn + IB) * sswn + 2 * kb + 1)
        e = lambda v: tl.reshape(tl.broadcast_to(v[:, None], (NB, 128)), (BLOCK_N,))
        ag += tl.dot(a0, tl.trans(g0)) * (sa0[:, None] * e(sg0)[None, :])
        ag += tl.dot(a1, tl.trans(g1)) * (sa1[:, None] * e(sg1)[None, :])
        au += tl.dot(a0, tl.trans(u0)) * (sa0[:, None] * e(su0)[None, :])
        au += tl.dot(a1, tl.trans(u1)) * (sa1[:, None] * e(su1)[None, :])
        ap += 256
        gp += 256
        up += 256
    g = ag.to(tl.bfloat16).to(tl.float32)
    u = au.to(tl.bfloat16).to(tl.float32)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    y = (s * u).to(tl.bfloat16).to(tl.float32)
    yb = tl.reshape(y, (BLOCK_M, NB, 128))
    sc = tl.maximum(tl.max(tl.abs(yb), axis=2) * RMAX, 1e-12)
    sce = tl.reshape(tl.broadcast_to(sc[:, :, None], (BLOCK_M, NB, 128)), (BLOCK_M, BLOCK_N))
    v = tl.minimum(tl.maximum(y / sce, -EMAX), EMAX)
    tl.store(Q + rm[:, None] * sqm + rn[None, :], v.to(FP8), mask=mm)
    tl.store(S + rm[:, None] * ssm + sn[None, :], sc, mask=rm[:, None] < M)


H, I = 3584, 2048
M = 4096
aq = torch.randint(0, 200, (M, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
asc = torch.rand(M, H // 128, device=dev) * 0.01
wq = torch.randint(0, 200, (2 * I, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
wsc = torch.rand(2 * I // 128, H // 128, device=dev) * 0.01
gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
gs = torch.empty((M, I // 128), device=dev)
FL = 2 * M * 2 * I * H

print("--- current _gemm1 ---")
for bm, bn, nw, ns in [(128, 128, 8, 2), (64, 128, 8, 3), (256, 128, 8, 2), (128, 256, 8, 2)]:
    t = bench(lambda: tk._gemm1[(triton.cdiv(M, bm), triton.cdiv(I, bn))](
        aq, asc, wq, wsc, gq, gs, M, I, aq.stride(0), asc.stride(0), wq.stride(0),
        wsc.stride(0), gq.stride(0), gs.stride(0), BLOCK_M=bm, BLOCK_N=bn,
        NUM_K=H // 128, num_warps=nw, num_stages=ns))
    print(f"  bm{bm} bn{bn} nw{nw} ns{ns}: {t*1e3:6.1f} us {FL/(t/1e3)/1e12:6.0f} TF/s")

print("--- noscale (upper bound) ---")
for bm, bn, nw, ns in [(128, 128, 8, 2), (256, 128, 8, 2), (128, 256, 8, 2), (128, 128, 8, 3)]:
    t = bench(lambda: _g1_noscale[(triton.cdiv(M, bm), triton.cdiv(I, bn))](
        aq, wq, gq, M, I, aq.stride(0), wq.stride(0), gq.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, NUM_K=H // 128, num_warps=nw, num_stages=ns))
    print(f"  bm{bm} bn{bn} nw{nw} ns{ns}: {t*1e3:6.1f} us {FL/(t/1e3)/1e12:6.0f} TF/s")

print("--- k256 ---")
for bm, bn, nw, ns in [(128, 128, 8, 2), (64, 128, 8, 3), (128, 128, 8, 1), (256, 128, 8, 2)]:
    t = bench(lambda: _g1_k256[(triton.cdiv(M, bm), triton.cdiv(I, bn))](
        aq, asc, wq, wsc, gq, gs, M, I, aq.stride(0), asc.stride(0), wq.stride(0),
        wsc.stride(0), gq.stride(0), gs.stride(0), BLOCK_M=bm, BLOCK_N=bn,
        NUM_K=H // 128, num_warps=nw, num_stages=ns))
    print(f"  bm{bm} bn{bn} nw{nw} ns{ns}: {t*1e3:6.1f} us {FL/(t/1e3)/1e12:6.0f} TF/s")
