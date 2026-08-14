import torch, inspect, traceback
import aiter

dev = torch.device("cuda:0")
H, V = 2048, 102400

for nm in ["gemm_a16w16_asm", "gemm_a16w16_opus"]:
    f = getattr(aiter, nm)
    try:
        print(nm, inspect.signature(f))
    except Exception as e:
        print(nm, "sig?", e)
    print("  doc:", (f.__doc__ or "")[:300])

def timeit(fn, iters=10, warmup=3, reps=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        st = torch.cuda.Event(True); en = torch.cuda.Event(True)
        st.record()
        for _ in range(iters): fn()
        en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) / iters * 1e3)
    return min(ts)

torch.manual_seed(0)
w = (torch.randn(V, H, dtype=torch.bfloat16, device=dev) * (1.0 / H**0.5))
wt = w.t()

for M in [128, 1024, 4096, 8192]:
    x = torch.randn(M, H, dtype=torch.bfloat16, device=dev)
    ref = torch.matmul(x, wt)
    tref = timeit(lambda: torch.matmul(x, wt))
    print(f"--- M={M} torch={tref:.1f}us")
    for nm in ["gemm_a16w16_asm", "gemm_a16w16_opus"]:
        f = getattr(aiter, nm)
        for variant in ["w", "wt"]:
            B = w if variant == "w" else wt
            for without in [True, False]:
                try:
                    if without:
                        o = torch.empty(M, V, dtype=torch.bfloat16, device=dev)
                        r = f(x, B, o)
                        call = lambda: f(x, B, o)
                    else:
                        r = f(x, B)
                        call = lambda: f(x, B)
                    torch.cuda.synchronize()
                    rr = r if isinstance(r, torch.Tensor) else o
                    if rr.shape != ref.shape:
                        print(f"    {nm}/{variant}/out={without}: shape {tuple(rr.shape)}")
                        continue
                    err = (rr.float() - ref.float()).abs().max().item()
                    t = timeit(call)
                    print(f"    {nm}/{variant}/out={without}: {t:8.1f}us err={err:.4g} "
                          f"{2*M*H*V/t*1e-6:7.1f}TF")
                except Exception as e:
                    print(f"    {nm}/{variant}/out={without}: FAIL {type(e).__name__} {str(e)[:100]}")
    del x, ref; torch.cuda.empty_cache()
