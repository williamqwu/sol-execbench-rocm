import sys, torch, time, triton, itertools, json
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import tk5 as tk

dev = 'cuda'
torch.manual_seed(0)


def bench(fn, n=40, w=15):
    try:
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
    except Exception:
        return 1e9
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in ev)
    return ts[len(ts) // 2]


H, I = 3584, 2048
MS = [384, 640, 896, 1024, 1152, 1536, 1792, 1920, 2048, 2176, 2432, 2816,
      3072, 3584, 3712, 4096]
which = sys.argv[1]

aqW = torch.randint(0, 200, (2 * I, H), device=dev, dtype=torch.uint8)
wq = aqW.view(torch.float8_e4m3fn)
wsc = torch.rand(2 * I // 128, H // 128, device=dev) * 0.01
dq = torch.randint(0, 200, (H, I), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
dsc = torch.rand(H // 128, I // 128, device=dev) * 0.01

best = {}
for M in MS:
    aq = torch.randint(0, 200, (M, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    asc = torch.rand(M, H // 128, device=dev) * 0.01
    rw = torch.randn(M, 1, device=dev, dtype=torch.bfloat16)
    gq = torch.randint(0, 200, (M, I), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    gs = torch.rand(M, I // 128, device=dev) * 0.01
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    res = []
    if which == "g1":
        for bm, nw, ns, wpe in itertools.product([16, 32, 64, 128, 256], [4, 8], [1, 2, 3], [1, 2, 4]):
            if bm * 256 // (nw * 64) > 256 or bm * 256 < nw * 64 * 8:
                continue
            def f(bm=bm, nw=nw, ns=ns, wpe=wpe):
                tk._gemm1[(triton.cdiv(M, bm), I // 128)](
                    aq, asc, wq, wsc, gq, gs, M, aq.stride(0), asc.stride(0),
                    wq.stride(0), wsc.stride(0), gq.stride(0), gs.stride(0),
                    BLOCK_M=bm, NUM_K=H // 128, num_warps=nw, num_stages=ns,
                    waves_per_eu=wpe)
            res.append((bench(f), (bm, nw, ns, wpe)))
    else:
        for bm, bn, nw, ns, wpe in itertools.product(
                [16, 32, 64, 128, 256], [128, 256, 512], [4, 8], [1, 2, 3], [1, 2, 4]):
            if bm * bn // (nw * 64) > 256 or bm * bn < nw * 64 * 8:
                continue
            def f(bm=bm, bn=bn, nw=nw, ns=ns, wpe=wpe):
                tk._gemm2[(triton.cdiv(M, bm), triton.cdiv(H, bn))](
                    gq, gs, dq, dsc, rw, out, M, gq.stride(0), gs.stride(0),
                    dq.stride(0), dsc.stride(0), out.stride(0), BLOCK_M=bm,
                    BLOCK_N=bn, NUM_K=I // 128, num_warps=nw, num_stages=ns,
                    waves_per_eu=wpe)
            res.append((bench(f), (bm, bn, nw, ns, wpe)))
    res.sort()
    best[M] = res[0][1]
    print(M, [(round(t * 1e3, 1), c) for t, c in res[:4]], flush=True)

print(json.dumps({str(k): v for k, v in best.items()}))
