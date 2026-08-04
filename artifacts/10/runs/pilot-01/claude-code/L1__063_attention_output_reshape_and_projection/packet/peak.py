import time, torch

dev = "cuda:0"


def bench(fn, iters=20, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


print("peak bf16 GEMM probe")
for M, K, N in [(8192, 8192, 8192), (16384, 16384, 16384), (4096, 16384, 7168),
                (8192, 16384, 7168), (16384, 16384, 7168)]:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    t = bench(lambda: torch.matmul(a, b.t()))
    print(f"  {M}x{K}x{N}: {t:.3f} ms  {2*M*K*N/t*1e-9:7.0f} TFLOPS")
    del a, b
    torch.cuda.empty_cache()

print("\nblas library:", torch.backends.cuda.preferred_blas_library())
