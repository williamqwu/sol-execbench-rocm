import sys, torch, triton
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
MS = [384, 640, 896, 1024, 1152, 1536, 1792, 1920, 2048, 2176, 2432, 2816,
      3072, 3584, 3712, 4096]
CAND = [(128, 128, 8, 3, 2), (128, 128, 4, 2, 2), (128, 128, 8, 2, 2),
        (64, 128, 8, 2, 2), (64, 128, 8, 3, 2)]
dq = torch.randint(0, 200, (H, I), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
dsc = torch.rand(H // 128, I // 128, device=dev) * 0.01
for M in MS:
    rw = torch.randn(M, 1, device=dev, dtype=torch.bfloat16)
    gq = torch.randint(0, 200, (M, I), device=dev, dtype=torch.uint8).view(torch.float8_e4m3fn)
    gs = torch.rand(I // 128, M, device=dev) * 0.01
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    ts = []
    for bm, bn, nw, ns, wpe in CAND:
        ts.append(bench(lambda bm=bm, bn=bn, nw=nw, ns=ns, wpe=wpe: tk._gemm2[
            (triton.cdiv(M, bm), triton.cdiv(H, bn))](
            gq, gs, dq, dsc, rw, out, M, gq.stride(0), gs.stride(0),
            dq.stride(0), dsc.stride(0), out.stride(0), BLOCK_M=bm, BLOCK_N=bn,
            NUM_K=I // 128, num_warps=nw, num_stages=ns, waves_per_eu=wpe)))
    b = min(ts)
    print(f"M={M:5d} " + "  ".join(
        f"{c[0]}/{c[2]}/{c[3]}={t*1e3:5.1f}{'*' if t == b else ' '}"
        for c, t in zip(CAND, ts)), flush=True)
