import torch, sys
import tk, tk2

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

# finer N tiles so tile count lands closer to a multiple of 256 CUs
cfgs = []
for BM in (16, 32, 64, 128, 256):
    for BN in (64, 128, 160, 200, 256, 320, 400, 512):
        for BK in (64, 128, 256):
            if BM * BN * 4 > 256 * 1024:  # acc regs sanity
                continue
            if BM * BK * 2 + BK * BN * 2 > 160 * 1024:
                continue
            for nw in (4, 8):
                for ns in (2, 3, 4):
                    cfgs.append(dict(BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                     num_warps=nw, num_stages=ns))
print(f"{len(cfgs)} cfgs", flush=True)

for M in Ms:
    x = torch.randn(M, H, dtype=torch.bfloat16, device=dev)
    ref = torch.matmul(x, wt)
    tref = timeit(lambda: torch.matmul(x, wt))
    out = torch.empty((M, V), device=dev, dtype=torch.bfloat16)
    res = []
    for cfg in cfgs:
        for kind in ("persist", "grid"):
            try:
                if kind == "persist":
                    for occ in (1, 2):
                        c2 = dict(cfg, occ=occ)
                        tk2.persist(x, wt, c2, out)
                        torch.cuda.synchronize()
                        if (out.float() - ref.float()).abs().max().item() > 0.02: continue
                        t = timeit(lambda: tk2.persist(x, wt, c2, out))
                        if t: res.append((t, "persist", c2))
                else:
                    cg = dict(cfg, GROUP_M=8)
                    tk.gemm(x, wt, cg, out)
                    torch.cuda.synchronize()
                    if (out.float() - ref.float()).abs().max().item() > 0.02: continue
                    t = timeit(lambda: tk.gemm(x, wt, cg, out))
                    if t: res.append((t, "grid", cg))
            except Exception:
                continue
    res.sort(key=lambda r: r[0])
    print(f"### M={M} torch={tref:.1f}us")
    for t, kind, cfg in res[:8]:
        c = {k: v for k, v in cfg.items() if k != "GROUP_M"}
        print(f"  {t:8.1f}us {2*M*H*V/t*1e-6:7.1f}TF {kind:8s} {c}")
    sys.stdout.flush()
    del x, ref, out; torch.cuda.empty_cache()
