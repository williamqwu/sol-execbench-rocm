import sys, torch, time, triton, itertools
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import tk3, tk4

dev = 'cuda'


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
for M in [384, 1024, 2048, 4096]:
    aq = torch.randint(0, 200, (M, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    asc = torch.rand(M, H // 128, device=dev) * 0.01
    wq = torch.randint(0, 200, (2 * I, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    wsc = torch.rand(2 * I // 128, H // 128, device=dev) * 0.01
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((M, I // 128), device=dev)
    FL = 2 * M * 2 * I * H
    r3, r4 = [], []
    for bm, bn, nw, ns, wpe in itertools.product([64, 128, 256], [128, 256], [4, 8], [1, 2, 3], [1, 2]):
        if bm * bn * 2 // (nw * 64) > 256:
            continue
        t = bench(lambda: tk3._gemm1[(triton.cdiv(M, bm), triton.cdiv(I, bn))](
            aq, asc, wq, wsc, gq, gs, M, I, aq.stride(0), asc.stride(0), wq.stride(0),
            wsc.stride(0), gq.stride(0), gs.stride(0), BLOCK_M=bm, BLOCK_N=bn,
            NUM_K=H // 128, num_warps=nw, num_stages=ns, waves_per_eu=wpe))
        r3.append((t, (bm, bn, nw, ns, wpe)))
    for bm, nw, ns, wpe in itertools.product([32, 64, 128, 256], [4, 8], [1, 2, 3], [1, 2]):
        if bm * 256 // (nw * 64) > 256:
            continue
        t = bench(lambda: tk4._gemm1[(triton.cdiv(M, bm), I // 128)](
            aq, asc, wq, wsc, gq, gs, M, aq.stride(0), asc.stride(0), wq.stride(0),
            wsc.stride(0), gq.stride(0), gs.stride(0), BLOCK_M=bm, NUM_K=H // 128,
            num_warps=nw, num_stages=ns, waves_per_eu=wpe))
        r4.append((t, (bm, nw, ns, wpe)))
    r3.sort(); r4.sort()
    print(f"M={M}")
    print(f"  tk3(dual acc) best {r3[0][1]}: {r3[0][0]*1e3:6.1f} us {FL/(r3[0][0]/1e3)/1e12:5.0f} TF")
    print(f"  tk4(interleave) best {r4[0][1]}: {r4[0][0]*1e3:6.1f} us {FL/(r4[0][0]/1e3)/1e12:5.0f} TF")
    print("   tk4 top3:", [(round(t * 1e3, 1), c) for t, c in r4[:3]])
