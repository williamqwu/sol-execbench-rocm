import torch, itertools, sys, json, traceback
import tk

H, V = 2048, 102400
dev = torch.device("cuda:0")
torch.manual_seed(0)
w = (torch.randn(V, H, dtype=torch.bfloat16, device=dev) * (1.0 / H**0.5))
wt = w.t()

def timeit(fn, iters=10, warmup=3, reps=3):
    try:
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
    except Exception:
        return None
    ts = []
    for _ in range(reps):
        st = torch.cuda.Event(True); en = torch.cuda.Event(True)
        st.record()
        for _ in range(iters): fn()
        en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) / iters * 1e3)
    return min(ts)

Ms = [int(x) for x in sys.argv[1].split(",")]

cfgs = []
for BM, BN, BK in [
    (64, 256, 64), (64, 256, 128), (128, 256, 64), (128, 256, 128),
    (256, 256, 64), (128, 128, 64), (128, 128, 128), (256, 128, 64),
    (64, 128, 128), (64, 128, 64), (32, 256, 128), (32, 256, 64),
    (16, 256, 128), (16, 512, 128), (32, 512, 128), (64, 512, 64),
    (128, 512, 64), (16, 128, 256), (32, 128, 256), (64, 64, 256),
]:
    for nw in (4, 8):
        for ns in (2, 3, 4):
            for gm in (1, 8):
                cfgs.append(dict(BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=gm,
                                 num_warps=nw, num_stages=ns))

for M in Ms:
    x = torch.randn(M, H, dtype=torch.bfloat16, device=dev)
    ref = torch.matmul(x, wt)
    tref = timeit(lambda: torch.matmul(x, wt))
    out = torch.empty((M, V), device=dev, dtype=torch.bfloat16)
    best = []
    for cfg in cfgs:
        try:
            tk.gemm(x, wt, cfg, out)
            torch.cuda.synchronize()
            if not torch.equal(out, ref):
                d = (out.float() - ref.float()).abs().max().item()
                if d > 0.02: continue
            t = timeit(lambda: tk.gemm(x, wt, cfg, out))
            if t: best.append((t, cfg))
        except Exception:
            continue
    best.sort(key=lambda r: r[0])
    print(f"### M={M} torch={tref:.1f}us")
    for t, cfg in best[:6]:
        print(f"  {t:8.1f}us  {2*M*H*V/t*1e-6:7.1f} TF  {cfg}")
    sys.stdout.flush()
    del x, ref, out
    torch.cuda.empty_cache()
