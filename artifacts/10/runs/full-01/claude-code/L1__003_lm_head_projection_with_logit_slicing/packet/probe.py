import torch, time

dev = torch.device("cuda:0")

def timeit(fn, iters=20, warmup=5, reps=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        st = torch.cuda.Event(True); en = torch.cuda.Event(True)
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) / iters * 1e3)
    return min(ts)

# ---- peak flops: large square GEMMs, various layouts
print("=== peak bf16 flops (square GEMM) ===")
for N in [4096, 8192, 16384]:
    a = torch.randn(N, N, dtype=torch.bfloat16, device=dev)
    b = torch.randn(N, N, dtype=torch.bfloat16, device=dev)
    bt = b.t().contiguous().t()   # TN layout (K-contiguous B)
    for name, bb in [("NN", b), ("TN", bt)]:
        t = timeit(lambda: torch.matmul(a, bb), iters=10)
        print(f"  N={N} {name}: {t:8.1f} us  {2.0*N**3/t*1e-6:8.1f} TFLOP/s")
    del a, b, bt
    torch.cuda.empty_cache()

# ---- peak read bandwidth
print("=== peak bandwidth ===")
big = torch.empty(1024*1024*1024, dtype=torch.uint8, device=dev)  # 1 GiB
t = timeit(lambda: big.sum(), iters=10)
print(f"  read 1GiB sum: {t:8.1f} us -> {big.numel()/t*1e-6:8.2f} TB/s")
dst = torch.empty_like(big)
t = timeit(lambda: dst.copy_(big), iters=10)
print(f"  copy 1GiB:     {t:8.1f} us -> {2*big.numel()/t*1e-6:8.2f} TB/s (r+w)")
del big, dst
torch.cuda.empty_cache()

# ---- the actual problem: write-only cost of the output
print("=== output write cost ===")
for M in [128, 1024, 2048, 4096, 8192]:
    o = torch.empty(M, 102400, dtype=torch.bfloat16, device=dev)
    t = timeit(lambda: o.zero_(), iters=10)
    print(f"  M={M:5d} zero {o.numel()*2/1e6:8.1f}MB: {t:8.1f} us -> {o.numel()*2/t*1e-6:6.2f} TB/s")
    del o
    torch.cuda.empty_cache()
