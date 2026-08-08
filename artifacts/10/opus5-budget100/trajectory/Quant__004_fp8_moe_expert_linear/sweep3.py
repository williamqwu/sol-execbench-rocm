import sys, torch, time, triton, itertools, json
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import tk3 as tk

dev = 'cuda'
torch.manual_seed(0)


def bench(fn, n=60, w=25):
    try:
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
    except Exception:
        return 1e9
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


H, I = 3584, 2048
MS = [384, 640, 896, 1024, 1152, 1536, 1792, 1920, 2048, 2176, 2432, 2816,
      3072, 3584, 3712, 4096]
which = sys.argv[1]

guw = torch.randn(2 * I, H, device=dev, dtype=torch.bfloat16) * H ** -0.5
dw = torch.randn(H, I, device=dev, dtype=torch.bfloat16) * I ** -0.5
wq = torch.empty((2 * I, H), dtype=torch.float8_e4m3fn, device=dev)
wsc = torch.empty((2 * I // 128, H // 128), dtype=torch.float32, device=dev)
dq = torch.empty((H, I), dtype=torch.float8_e4m3fn, device=dev)
dsc = torch.empty((H // 128, I // 128), dtype=torch.float32, device=dev)

best = {}
for M in MS:
    hs = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    rw = torch.randn(M, 1, device=dev, dtype=torch.bfloat16)
    aq = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=dev)
    asc = torch.empty((M, H // 128), dtype=torch.float32, device=dev)
    NP1 = (2 * I // 128) * (H // 128)
    NP2 = (H // 128) * (I // 128)
    tk._quant_all[(NP1 + NP2 + triton.cdiv(M, 64) * (H // 128),)](
        guw, wq, wsc, dw, dq, dsc, hs, aq, asc, M, guw.stride(0), wsc.stride(0),
        dw.stride(0), dsc.stride(0), hs.stride(0), aq.stride(0), asc.stride(0),
        KB1=H // 128, KB2=I // 128, KBA=H // 128, NP1=NP1, NP2=NP2, BM=64,
        num_warps=4)
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((M, I // 128), dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    res = []
    if which == "g1":
        for bm, bn, nw, ns, wpe in itertools.product(
                [16, 32, 64, 128, 256], [128, 256], [4, 8], [1, 2, 3], [1, 2, 4]):
            if bm * bn * 2 // (nw * 64) > 256 or bm * bn * 2 < nw * 64 * 16:
                continue
            def f(bm=bm, bn=bn, nw=nw, ns=ns, wpe=wpe):
                tk._gemm1[(triton.cdiv(M, bm), triton.cdiv(I, bn))](
                    aq, asc, wq, wsc, gq, gs, M, I, aq.stride(0), asc.stride(0),
                    wq.stride(0), wsc.stride(0), gq.stride(0), gs.stride(0),
                    BLOCK_M=bm, BLOCK_N=bn, NUM_K=H // 128, num_warps=nw,
                    num_stages=ns, waves_per_eu=wpe)
            res.append((bench(f), (bm, bn, nw, ns, wpe)))
    else:
        for bm, bn, nw, ns, wpe in itertools.product(
                [16, 32, 64, 128, 256], [128, 256, 512], [4, 8], [1, 2, 3], [1, 2, 4]):
            if bm * bn // (nw * 64) > 256 or bm * bn < nw * 64 * 16:
                continue
            def f(bm=bm, bn=bn, nw=nw, ns=ns, wpe=wpe):
                tk._gemm2[(triton.cdiv(M, bm), triton.cdiv(H, bn))](
                    gq, gs, dq, dsc, rw, out, M, gq.stride(0), gs.stride(0),
                    dq.stride(0), dsc.stride(0), out.stride(0), BLOCK_M=bm,
                    BLOCK_N=bn, NUM_K=I // 128, num_warps=nw, num_stages=ns,
                    waves_per_eu=wpe)
            res.append((bench(f), (bm, bn, nw, ns, wpe)))
    res.sort()
    best[M] = res[0]
    print(M, [(round(t * 1e3, 2), c) for t, c in res[:5]], flush=True)

print(json.dumps({str(k): v[1] for k, v in best.items()}))
