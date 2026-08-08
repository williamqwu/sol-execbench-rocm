import sys, torch, time, triton, itertools, json
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import reference as R
import tk

dev = 'cuda'
torch.manual_seed(0)


def bench(fn, n=50, w=20):
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

which = sys.argv[1] if len(sys.argv) > 1 else "g1"

guw = torch.randn(2 * I, H, device=dev, dtype=torch.bfloat16) * H ** -0.5
dw = torch.randn(H, I, device=dev, dtype=torch.bfloat16) * I ** -0.5
wq, wsc = tk.quant_weight(guw)
dq, dsc = tk.quant_weight(dw)

best = {}
for M in MS:
    hs = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    rw = torch.randn(M, 1, device=dev, dtype=torch.bfloat16)
    aq, asc = tk.quant_act(hs)
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((M, I // 128), dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    res = []
    if which == "g1":
        for bm, nw, ns, kp in itertools.product([32, 64, 128, 256], [4, 8], [1, 2, 3], [1, 2]):
            if bm * 128 * 2 // (nw * 64) > 256:
                continue
            def f(bm=bm, nw=nw, ns=ns, kp=kp):
                tk._gemm1_silu_quant[(triton.cdiv(M, bm), I // 128)](
                    aq, asc, wq, wsc, gq, gs, M, H, I, aq.stride(0), asc.stride(0),
                    wq.stride(0), wsc.stride(0), gq.stride(0), gs.stride(0),
                    BLOCK_M=bm, NUM_K=H // 128, num_warps=nw, num_stages=ns,
                    waves_per_eu=kp)
            res.append((bench(f), (bm, nw, ns, kp)))
    else:
        for bm, bn, nw, ns, kp in itertools.product([32, 64, 128, 256], [128, 256], [4, 8], [1, 2, 3], [1, 2]):
            if bm * bn // (nw * 64) > 256:
                continue
            def f(bm=bm, bn=bn, nw=nw, ns=ns, kp=kp):
                tk._gemm2[(triton.cdiv(M, bm), H // bn)](
                    gq, gs, dq, dsc, rw, out, M, I, H, gq.stride(0), gs.stride(0),
                    dq.stride(0), dsc.stride(0), out.stride(0),
                    BLOCK_M=bm, BLOCK_N=bn, NUM_K=I // 128, num_warps=nw,
                    num_stages=ns, waves_per_eu=kp)
            res.append((bench(f), (bm, bn, nw, ns, kp)))
    res.sort()
    best[M] = res[0]
    print(M, [(round(t, 4), c) for t, c in res[:4]], flush=True)

print(json.dumps({str(k): v for k, v in best.items()}))
