import torch, time

H, V = 2048, 102400
Ms = [128, 256, 293, 691, 1024, 2048, 3011, 3412, 3988, 4096, 8192]

dev = torch.device("cuda:0")
torch.manual_seed(0)
w = (torch.randn(V, H, dtype=torch.bfloat16, device=dev) * (1.0 / H**0.5))
wt = w.t()

def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        st = torch.cuda.Event(True); en = torch.cuda.Event(True)
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) / iters * 1e3)  # us
    return min(ts)

PEAK_FLOPS = 2.5e15
PEAK_BW = 8.0e12

print(f"{'M':>6} {'ref_us':>9} {'TFLOPs':>8} {'GB/s':>8} {'SOL_us':>8} {'eff%':>6}")
for M in Ms:
    x = torch.randn(M, H, dtype=torch.bfloat16, device=dev)
    x3 = x.view(1, M, H)
    f = lambda: torch.matmul(x3, wt)
    t = timeit(f)
    flops = 2.0 * M * H * V
    byts = (M * H + V * H + M * V) * 2
    sol = max(flops / PEAK_FLOPS, byts / PEAK_BW) * 1e6
    print(f"{M:>6} {t:>9.1f} {flops/t*1e-6:>8.1f} {byts/t*1e-3:>8.1f} {sol:>8.1f} {sol/t*100:>6.1f}")
    del x, x3
    torch.cuda.empty_cache()
