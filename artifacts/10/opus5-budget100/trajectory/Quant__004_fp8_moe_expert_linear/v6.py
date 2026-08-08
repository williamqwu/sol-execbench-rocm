import sys, torch, time, triton, triton.language as tl, itertools
dev = 'cuda'
FP8 = tl.float8e4nv


def bench(fn, n=60, w=25):
    try:
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
    except Exception as e:
        print("ERR", str(e)[:200])
        return 1e9
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


# MODE 0: scalar sg (BLOCK_N=128), fused rowvec = sa*sg  -> 1 fma/elem
# MODE 1: same but explicit tl.math.fma
# MODE 2: reshape-broadcast (what tk3 does)
# MODE 3: no scale (bound)
@triton.jit
def k1(A, SA, W, SW, C, M, sam, ssam, swn, sswn, scm,
       BLOCK_M: tl.constexpr, NUM_K: tl.constexpr, MODE: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * 128 + tl.arange(0, 128)
    rk = tl.arange(0, 128)
    ap = A + rm[:, None] * sam + rk[None, :]
    bp = W + rn[:, None] * swn + rk[None, :]
    mm = rm[:, None] < M
    acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    for kb in tl.range(0, NUM_K):
        a = tl.load(ap, mask=mm, other=0.0)
        b = tl.load(bp)
        if MODE == 3:
            acc = tl.dot(a, tl.trans(b), acc)
        else:
            sa = tl.load(SA + rm * ssam + kb, mask=rm < M, other=0.0)
            sg = tl.load(SW + pn * sswn + kb)
            if MODE == 0:
                t = sa * sg
                acc += tl.dot(a, tl.trans(b)) * t[:, None]
            elif MODE == 1:
                t = sa * sg
                acc = tl.math.fma(tl.dot(a, tl.trans(b)), t[:, None], acc)
            else:
                sbe = tl.reshape(tl.broadcast_to(sg[None, None], (1, 128)), (128,))
                acc += tl.dot(a, tl.trans(b)) * (sa[:, None] * sbe[None, :])
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
nm = {0: "scalar sg, fused rowvec", 1: "explicit fma", 2: "reshape bcast", 3: "no scale"}
for mode in [3, 0, 1, 2]:
    res = []
    for bm, nw, ns, wpe in itertools.product([64, 128, 256], [4, 8], [1, 2, 3], [1, 2]):
        if bm * 128 // (nw * 64) > 256:
            continue
        t = bench(lambda: k1[(triton.cdiv(M, bm), N // 128)](
            aq, asc, wq, wsc, C, M, aq.stride(0), asc.stride(0), wq.stride(0),
            wsc.stride(0), C.stride(0), BLOCK_M=bm, NUM_K=H // 128, MODE=mode,
            num_warps=nw, num_stages=ns, waves_per_eu=wpe))
        res.append((t, (bm, nw, ns, wpe)))
    res.sort()
    print(f"MODE{mode} {nm[mode]:24s}: " +
          " | ".join(f"{c} {t*1e3:.1f}us {FL/(t/1e3)/1e12:.0f}TF" for t, c in res[:3]))
