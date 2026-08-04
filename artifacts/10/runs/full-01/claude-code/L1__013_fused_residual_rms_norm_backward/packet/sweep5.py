import torch, triton
import triton.language as tl
from variants import bench_ev, mk, DEV, HID

# How much of small-M time is CPU-side (python + launch) vs GPU?

@triton.jit
def _nop(X):
    pass

x1 = torch.empty(1, device=DEV)

print("=== launch-overhead microbenchmarks (per call, us) ===")
t = bench_ev(lambda: None, (), iters=200)
print(f"  python no-op            : {t*1000:7.2f}")
t = bench_ev(lambda: _nop[(1,)](x1), (), iters=200)
print(f"  1 empty triton launch   : {t*1000:7.2f}")
t = bench_ev(lambda: (_nop[(1,)](x1), _nop[(1,)](x1)), (), iters=200)
print(f"  2 empty triton launches : {t*1000:7.2f}")

for M in (128, 1546, 8192):
    go = torch.empty(M, HID, device=DEV, dtype=torch.bfloat16)
    t = bench_ev(lambda: torch.empty_like(go), (), iters=200)
    print(f"  empty_like M={M:6d}      : {t*1000:7.2f}")
    t = bench_ev(lambda: torch.zeros(HID, device=DEV, dtype=torch.float32), (), iters=200)
    print(f"  torch.zeros(2560)       : {t*1000:7.2f}")
    break

for P in (64, 128, 256, 512):
    part = torch.empty((P, HID), device=DEV, dtype=torch.float32)
    t = bench_ev(lambda: part.sum(0), (), iters=200)
    print(f"  torch part.sum(0) P={P:4d}: {t*1000:7.2f}")

# CUDA-graph pure-GPU time for the whole run, small M
from sweep4 import make
print("\n=== eager vs cudagraph (isolates CPU overhead) ===")
for M in (128, 422, 1249, 1546, 3412, 8192, 15952):
    args = mk(M)
    f = make(2, lambda m: min(m, 512), 4)
    t_eager = bench_ev(f, args, iters=50)
    # graph capture
    try:
        g = torch.cuda.CUDAGraph()
        f(*args)
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                f(*args)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            f(*args)
        torch.cuda.synchronize()
        t_g = bench_ev(lambda: g.replay(), (), iters=50)
    except Exception as e:
        t_g = float('nan')
        print("   graph fail", e)
    print(f"  M={M:7d}  eager={t_eager*1000:8.1f}us  graph(GPU only)={t_g*1000:8.1f}us  cpu_overhead={(t_eager-t_g)*1000:7.1f}us")
    del args
    torch.cuda.empty_cache()
