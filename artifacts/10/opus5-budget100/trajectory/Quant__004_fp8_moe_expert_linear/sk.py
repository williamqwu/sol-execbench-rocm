import sys, torch, triton, itertools
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import tk6 as tk
dev = 'cuda'


def bench(fn, n=60, w=25):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in ev)
    return ts[len(ts) // 2]


H, I = 3584, 2048
KB1 = H // 128
wq = torch.randint(0, 200, (2 * I, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
wsc = torch.rand(2 * I // 128, KB1, device=dev) * 0.01
print("gemm1 with the k-loop shortened to NUM_K=KB1/SK and grid x SK:")
print("(upper bound on what split-K could reach, ignoring the reduce pass)")
for M in [384, 640, 896, 1024, 1152]:
    aq = torch.randint(0, 200, (M, H), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    asc = torch.rand(KB1, M, device=dev) * 0.01
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((I // 128, M), device=dev)
    out = []
    for sk in [1, 2, 4, 7]:
        nk = KB1 // sk
        best = 1e9
        for bm, nw, ns, wpe in itertools.product([64, 128], [8], [2, 3], [2]):
            g = (triton.cdiv(M, bm) * sk, I // 128)
            t = bench(lambda bm=bm, nw=nw, ns=ns, wpe=wpe, nk=nk, g=g: tk._gemm1[g](
                aq, asc, wq, wsc, gq, gs, M, aq.stride(0), asc.stride(0),
                wq.stride(0), wsc.stride(0), gq.stride(0), gs.stride(0),
                BLOCK_M=bm, NUM_K=nk, num_warps=nw, num_stages=ns,
                waves_per_eu=wpe))
            best = min(best, t)
        out.append((sk, best))
    print(f"M={M:5d} " + "  ".join(f"SK{s}={t*1e3:5.1f}us" for s, t in out),
          flush=True)
