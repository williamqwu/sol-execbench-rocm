import torch, sys, csv, traceback
import aiter

H, V = 2048, 102400
dev = torch.device("cuda:0")
torch.manual_seed(0)
w = (torch.randn(V, H, dtype=torch.bfloat16, device=dev) * (1.0 / H**0.5))
wt = w.t()

rows = list(csv.DictReader(open("/sgl-workspace/aiter/hsa/gfx950/bf16gemm/bf16gemm_fp32bf16.csv")))
knls = [(r["knl_name"], int(r["tileM"]), int(r["tileN"]), int(r["splitK"]),
         int(r["bPreshuffle"])) for r in rows]
print(f"{len(knls)} kernels in csv")
for k in knls: print("  ", k)

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

WL = [int(x) for x in sys.argv[1].split(",")]

for M in WL:
    x = torch.randn(M, H, dtype=torch.bfloat16, device=dev)
    ref = torch.matmul(x, wt)
    tref = timeit(lambda: torch.mm(x, wt))
    out = torch.empty(M, V, dtype=torch.bfloat16, device=dev)
    res = []
    # default dispatch
    try:
        aiter.gemm_a16w16_asm(x, w, out); torch.cuda.synchronize()
        e = (out.float() - ref.float()).abs().max().item()
        if e < 0.02:
            t = timeit(lambda: aiter.gemm_a16w16_asm(x, w, out))
            if t: res.append((t, "default", None))
    except Exception as e:
        print("  default fail", str(e)[:120])
    # explicit kernels x splitK
    for kn, tm, tn, sk_flag, bps in knls:
        if bps: continue   # needs preshuffled weights
        sks = [None, 1, 2, 4, 8, 16] if sk_flag else [None]
        for sk in sks:
            try:
                out.zero_()
                aiter.gemm_a16w16_asm(x, w, out, None, sk, kn, False)
                torch.cuda.synchronize()
                e = (out.float() - ref.float()).abs().max().item()
                if e > 0.02: continue
                t = timeit(lambda: aiter.gemm_a16w16_asm(x, w, out, None, sk, kn, False))
                if t: res.append((t, kn.split("aiter")[-1][:40], sk))
            except Exception:
                continue
    res.sort(key=lambda r: r[0])
    print(f"### M={M} torch={tref:.1f}us ({2*M*H*V/tref*1e-6:.0f}TF)")
    for t, kn, sk in res[:6]:
        print(f"  {t:8.1f}us {2*M*H*V/t*1e-6:7.1f}TF  {kn} splitK={sk}")
    sys.stdout.flush()
    del x, ref, out; torch.cuda.empty_cache()
