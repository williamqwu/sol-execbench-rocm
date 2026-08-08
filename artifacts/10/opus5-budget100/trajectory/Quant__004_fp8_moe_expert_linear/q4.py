import sys, torch, triton, itertools
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import tk6 as tk
dev = 'cuda'


def bench(fn, n=60, w=25):
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
guw = torch.randn(2 * I, H, device=dev, dtype=torch.bfloat16)
dw = torch.randn(H, I, device=dev, dtype=torch.bfloat16)
wq = torch.empty((2 * I, H), dtype=torch.float8_e4m3fn, device=dev)
wsc = torch.empty((2 * I // 128, H // 128), device=dev)
dq = torch.empty((H, I), dtype=torch.float8_e4m3fn, device=dev)
dsc = torch.empty((H // 128, I // 128), device=dev)
KB1, KB2 = H // 128, I // 128
NP1 = (2 * I // 128) * KB1
NP2 = (H // 128) * KB2
for M in [384, 1024, 2048, 4096]:
    hs = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    aq = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=dev)
    asc = torch.empty((KB1, M), device=dev)
    r = []
    for bm, nw in itertools.product([16, 32, 64, 128, 256], [1, 2, 4, 8]):
        NPA = triton.cdiv(M, bm) * KB1
        t = bench(lambda bm=bm, nw=nw, NPA=NPA: tk._quant_all[(NP1 + NP2 + NPA,)](
            guw, wq, wsc, dw, dq, dsc, hs, aq, asc, M, guw.stride(0),
            wsc.stride(0), dw.stride(0), dsc.stride(0), hs.stride(0),
            aq.stride(0), asc.stride(0), IB=KB2, KB1=KB1, KB2=KB2, NP1=NP1,
            NP2=NP2, BM=bm, num_warps=nw))
        r.append((t, (bm, nw)))
    r.sort()
    print(f"M={M:5d}: " + " | ".join(f"BM{c[0]}/nw{c[1]} {t*1e3:.1f}us"
                                    for t, c in r[:5]), flush=True)
