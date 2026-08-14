import torch, os
H, V = 2048, 102400
dev = torch.device("cuda:0")
torch.manual_seed(0)
w = (torch.randn(V, H, dtype=torch.bfloat16, device=dev) * (1.0 / H**0.5))
wt = w.t()

def timeit(fn, iters=8, warmup=3, reps=3):
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

WL = [128, 293, 1024, 4096, 8192]
print("blas lib:", torch.backends.cuda.preferred_blas_library())

for lib in ["hipblaslt", "blaslt", "rocblas", "default"]:
    try:
        torch.backends.cuda.preferred_blas_library(lib)
    except Exception as e:
        print(f"{lib}: unsupported ({str(e)[:80]})")
        continue
    line = [f"{lib:10s}"]
    for M in WL:
        x = torch.randn(M, H, dtype=torch.bfloat16, device=dev)
        o = torch.empty(M, V, dtype=torch.bfloat16, device=dev)
        try:
            t = timeit(lambda: torch.mm(x, wt, out=o))
            line.append(f"M{M}={t:7.1f}")
        except Exception as e:
            line.append(f"M{M}=ERR")
        del x, o; torch.cuda.empty_cache()
    print(" ".join(line), flush=True)

torch.backends.cuda.preferred_blas_library("hipblaslt")

# torch.compile max-autotune
print("\n=== torch.compile max-autotune ===")
try:
    f = torch.compile(lambda a, b: torch.matmul(a, b), mode="max-autotune-no-cudagraphs",
                      dynamic=False)
    for M in [128, 4096]:
        x = torch.randn(M, H, dtype=torch.bfloat16, device=dev)
        ref = torch.matmul(x, wt)
        r = f(x, wt)
        err = (r.float() - ref.float()).abs().max().item()
        t = timeit(lambda: f(x, wt))
        print(f"  M={M}: {t:8.1f}us err={err:.4g}", flush=True)
        del x, ref, r; torch.cuda.empty_cache()
except Exception as e:
    print("  compile failed:", str(e)[:300])
