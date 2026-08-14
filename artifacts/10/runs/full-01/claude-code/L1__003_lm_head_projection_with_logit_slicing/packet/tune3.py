import torch, sys
import tk, tk2

H, V = 2048, 102400
dev = torch.device("cuda:0")
torch.manual_seed(0)
w = (torch.randn(V, H, dtype=torch.bfloat16, device=dev) * (1.0 / H**0.5))
wt = w.t()

def timeit(fn, iters=8, warmup=3, reps=3):
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

SHAPES = [
    (256, 256, 64), (256, 128, 64), (128, 256, 64), (128, 256, 128),
    (256, 256, 32), (128, 128, 64), (256, 64, 64), (64, 256, 128),
    (128, 512, 64), (256, 128, 128),
]
cfgs = []
for BM, BN, BK in SHAPES:
    for nw in (4, 8):
        for ns in (2, 3):
            for mi in (16, 32):
                for wpe in (1, 2):
                    cfgs.append(dict(BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                     num_warps=nw, num_stages=ns,
                                     matrix_instr_nonkdim=mi, waves_per_eu=wpe))
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
                    tk2.persist(x, wt, cfg, out)
                    torch.cuda.synchronize()
                    if (out.float() - ref.float()).abs().max().item() > 0.02: continue
                    t = timeit(lambda: tk2.persist(x, wt, cfg, out))
                else:
                    cg = dict(cfg, GROUP_M=8)
                    tk.gemm(x, wt, cg, out)
                    torch.cuda.synchronize()
                    if (out.float() - ref.float()).abs().max().item() > 0.02: continue
                    t = timeit(lambda: tk.gemm(x, wt, cg, out))
                if t: res.append((t, kind, cfg))
            except Exception:
                continue
    res.sort(key=lambda r: r[0])
    print(f"### M={M} torch={tref:.1f}us ({2*M*H*V/tref*1e-6:.0f}TF)")
    for t, kind, cfg in res[:6]:
        print(f"  {t:8.1f}us {2*M*H*V/t*1e-6:7.1f}TF {kind:8s} "
              f"BM{cfg['BLOCK_M']} BN{cfg['BLOCK_N']} BK{cfg['BLOCK_K']} "
              f"w{cfg['num_warps']} s{cfg['num_stages']} mi{cfg['matrix_instr_nonkdim']} "
              f"wpe{cfg['waves_per_eu']}")
    sys.stdout.flush()
    del x, ref, out; torch.cuda.empty_cache()
