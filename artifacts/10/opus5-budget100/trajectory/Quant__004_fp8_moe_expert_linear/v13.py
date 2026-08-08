import sys, torch, triton, itertools
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import tk5 as tk
dev = 'cuda'


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
for M in [2048, 4096]:
    aq = torch.randint(0, 200, (M, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    asc = torch.rand(M, H // 128, device=dev) * 0.01
    wq = torch.randint(0, 200, (2 * I, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    wsc = torch.rand(2 * I // 128, H // 128, device=dev) * 0.01
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((M, I // 128), device=dev)
    FL = 2 * M * 2 * I * H
    r = []
    for bm, nw, ns, wpe in itertools.product(
            [32, 64, 128], [4, 8], [1, 2], [1, 2, 3, 4, 6, 8]):
        if bm * 256 // (nw * 64) > 256 or bm * 256 < nw * 64 * 4:
            continue
        try:
            k = tk._gemm1[(triton.cdiv(M, bm), I // 128)](
                aq, asc, wq, wsc, gq, gs, M, aq.stride(0), asc.stride(0),
                wq.stride(0), wsc.stride(0), gq.stride(0), gs.stride(0),
                BLOCK_M=bm, NUM_K=H // 128, num_warps=nw, num_stages=ns,
                waves_per_eu=wpe)
        except Exception:
            continue
        t = bench(lambda: tk._gemm1[(triton.cdiv(M, bm), I // 128)](
            aq, asc, wq, wsc, gq, gs, M, aq.stride(0), asc.stride(0),
            wq.stride(0), wsc.stride(0), gq.stride(0), gs.stride(0),
            BLOCK_M=bm, NUM_K=H // 128, num_warps=nw, num_stages=ns,
            waves_per_eu=wpe))
        r.append((t, (bm, nw, ns, wpe), k.n_regs, k.n_spills))
    r.sort()
    print(f"M={M}: " + "\n       ".join(
        f"{c} {t*1e3:.1f}us {FL/(t/1e3)/1e12:.0f}TF regs={g} sp={s}"
        for t, c, g, s in r[:8]), flush=True)
