import time
import torch

DEV = "cuda:0"
N, K = 256, 7168


def graph_time(fn, iters=100, reps=5):
    """True GPU time per iteration, CPU dispatch removed via graph replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    best = 1e18
    for _ in range(reps):
        e0 = torch.cuda.Event(True)
        e1 = torch.cuda.Event(True)
        torch.cuda.synchronize()
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / iters * 1e3)
    return best


B = torch.randn(N, K, device=DEV, dtype=torch.float16)
Bt = B.T
print("  M     GPU-only us   bytes    eff GB/s")
for M in [1, 4, 16, 53, 80, 901, 11948, 14104]:
    A = torch.randn(M, K, device=DEV, dtype=torch.float16)
    t = graph_time(lambda: torch.matmul(A, Bt))
    byt = M * K * 2 + N * K * 2 + M * N * 2
    print(f"{M:6d}  {t:9.2f}   {byt/1e6:8.2f}MB  {byt/(t*1e-6)/1e9:8.0f}")
