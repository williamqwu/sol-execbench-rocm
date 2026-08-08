import sys, torch, triton, itertools
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import tk6 as tk
dev = 'cuda'


def bench(fn, n=50, w=20):
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
wq = torch.randint(0, 200, (2 * I, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
wsc = torch.rand(2 * I // 128, H // 128, device=dev) * 0.01
dq = torch.randint(0, 200, (H, I), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
dsc = torch.rand(H // 128, I // 128, device=dev) * 0.01
for M in MS:
    aq = torch.randint(0, 200, (M, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    asc = torch.rand(H // 128, M, device=dev) * 0.01
    rw = torch.randn(M, 1, device=dev, dtype=torch.bfloat16)
    gq = torch.randint(0, 200, (M, I), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    gs = torch.rand(I // 128, M, device=dev) * 0.01
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    row = []
    if which == "g1":
        for bm in [32, 64, 128, 256]:
            r = []
            for nw, ns, wpe in itertools.product([4, 8], [1, 2, 3], [1, 2]):
                if bm * 256 // (nw * 64) > 256 or bm * 256 < nw * 64 * 4:
                    continue
                r.append(bench(lambda bm=bm, nw=nw, ns=ns, wpe=wpe: tk._gemm1[
                    (triton.cdiv(M, bm), I // 128)](
                    aq, asc, wq, wsc, gq, gs, M, aq.stride(0), asc.stride(0),
                    wq.stride(0), wsc.stride(0), gq.stride(0), gs.stride(0),
                    BLOCK_M=bm, NUM_K=H // 128, num_warps=nw, num_stages=ns,
                    waves_per_eu=wpe)))
            row.append((bm, min(r) if r else 1e9))
    else:
        for bm in [32, 64, 128, 256]:
            r = []
            for bn, nw, ns, wpe in itertools.product([128, 256], [4, 8], [1, 2, 3], [1, 2]):
                if bm * bn // (nw * 64) > 256 or bm * bn < nw * 64 * 4:
                    continue
                r.append(bench(lambda bm=bm, bn=bn, nw=nw, ns=ns, wpe=wpe: tk._gemm2[
                    (triton.cdiv(M, bm), triton.cdiv(H, bn))](
                    gq, gs, dq, dsc, rw, out, M, gq.stride(0), gs.stride(0),
                    dq.stride(0), dsc.stride(0), out.stride(0), BLOCK_M=bm,
                    BLOCK_N=bn, NUM_K=I // 128, num_warps=nw, num_stages=ns,
                    waves_per_eu=wpe)))
            row.append((bm, min(r) if r else 1e9))
    best = min(t for _, t in row)
    print(f"M={M:5d} " + "  ".join(
        f"bm{bm}={t*1e3:6.1f}{'*' if t == best else ' '}" for bm, t in row),
        flush=True)
