import torch, time
dev = 'cuda'


def bench(fn, n=50, w=20):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


for (M, N, K, tag) in [(4096, 4096, 3584, "g1"), (4096, 3584, 2048, "g2"),
                       (1024, 4096, 3584, "g1"), (1024, 3584, 2048, "g2"),
                       (384, 4096, 3584, "g1"), (384, 3584, 2048, "g2")]:
    a = torch.randn(M, K, device=dev).to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
    sa = torch.ones(M, 1, device=dev)
    sb = torch.ones(1, N, device=dev)
    t = bench(lambda: torch._scaled_mm(a, b.T, scale_a=sa, scale_b=sb,
                                       out_dtype=torch.bfloat16))
    print(f"{tag} {M}x{N}x{K}: {t*1e3:7.1f} us  {2*M*N*K/(t/1e3)/1e12:6.0f} TF/s")
