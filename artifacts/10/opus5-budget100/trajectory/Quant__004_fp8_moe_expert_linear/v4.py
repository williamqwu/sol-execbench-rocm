import sys, torch, time, triton, triton.language as tl, itertools
dev = 'cuda'
FP8 = tl.float8e4nv


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


# single-accumulator scaled gemm: C = A@W.T with 1x128 / 128x128 scales
@triton.jit
def sgemm(A, SA, W, SW, C, M, N, sam, ssam, swn, sswn, scm,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_K: tl.constexpr,
          SCALED: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 128)
    ap = A + rm[:, None] * sam + rk[None, :]
    bp = W + rn[:, None] * swn + rk[None, :]
    mm = rm[:, None] < M
    NB: tl.constexpr = BLOCK_N // 128
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in tl.range(0, NUM_K):
        a = tl.load(ap, mask=mm, other=0.0)
        b = tl.load(bp)
        if SCALED:
            sa = tl.load(SA + rm * ssam + kb, mask=rm < M, other=0.0)
            sb = tl.load(SW + (pn * NB + tl.arange(0, NB)) * sswn + kb)
            sbe = tl.reshape(tl.broadcast_to(sb[:, None], (NB, 128)), (BLOCK_N,))
            acc += tl.dot(a, tl.trans(b)) * (sa[:, None] * sbe[None, :])
        else:
            acc = tl.dot(a, tl.trans(b), acc)
        ap += 128
        bp += 128
    tl.store(C + rm[:, None] * scm + rn[None, :], acc.to(tl.bfloat16), mask=mm)


H, I, M = 3584, 2048, 4096
N = 2 * I
aq = torch.randint(0, 200, (M, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
asc = torch.rand(M, H // 128, device=dev) * 0.01
wq = torch.randint(0, 200, (N, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
wsc = torch.rand(N // 128, H // 128, device=dev) * 0.01
C = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
FL = 2 * M * N * H
res = []
for sc, bm, bn, nw, ns, wpe in itertools.product(
        [1, 0], [64, 128, 256], [64, 128, 256], [4, 8], [1, 2, 3], [1, 2]):
    if bm * bn // (nw * 64) > 256 or bm * bn < nw * 64 * 8:
        continue
    t = bench(lambda: sgemm[(triton.cdiv(M, bm), triton.cdiv(N, bn))](
        aq, asc, wq, wsc, C, M, N, aq.stride(0), asc.stride(0), wq.stride(0),
        wsc.stride(0), C.stride(0), BLOCK_M=bm, BLOCK_N=bn, NUM_K=H // 128,
        SCALED=sc, num_warps=nw, num_stages=ns, waves_per_eu=wpe))
    res.append((t, sc, (bm, bn, nw, ns, wpe)))
for s in [1, 0]:
    r = sorted([x for x in res if x[1] == s])[:6]
    print("SCALED" if s else "PLAIN")
    for t, _, c in r:
        print(f"   {c}: {t*1e3:6.1f} us  {FL/(t/1e3)/1e12:6.0f} TF/s")
