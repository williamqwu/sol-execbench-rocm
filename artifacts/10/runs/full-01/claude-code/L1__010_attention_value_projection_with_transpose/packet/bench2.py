import json, sys, time
import torch

sys.path.insert(0, ".")
import reference
import kernel as K

DEV = "cuda:0"


def gpu_time(fn, *a, iters=200):
    """GPU-only time via CUDA graph replay (removes launch overhead)."""
    for _ in range(5):
        fn(*a)
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn(*a)
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn(*a)
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters // 10):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (iters // 10 * 10) * 1e6  # us


def wall_time(fn, *a, iters=100):
    for _ in range(20):
        fn(*a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


torch.manual_seed(0)
w = torch.randn(1024, 5120, device=DEV, dtype=torch.bfloat16)
shapes = [(1, 128), (1, 512), (8, 128), (1, 1571), (16, 128), (4, 541), (32, 128), (64, 128), (1, 8192)]
print(f"{'shape':>14} {'ref_gpu':>9} {'ref_wall':>9} {'my_gpu':>9} {'my_wall':>9}")
for (b, s) in shapes:
    h = torch.randn(b, s, 5120, device=DEV, dtype=torch.bfloat16)
    rg = gpu_time(reference.run, h, w)
    rw = wall_time(reference.run, h, w)
    mg = gpu_time(K.run, h, w)
    mw = wall_time(K.run, h, w)
    print(f"B={b:3d} S={s:5d} {rg:8.1f}u {rw:8.1f}u {mg:8.1f}u {mw:8.1f}u")
    del h
    torch.cuda.empty_cache()
