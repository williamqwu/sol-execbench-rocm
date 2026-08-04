import time
import torch
import triton
import triton.language as tl

DEV = "cuda:0"
N, K = 256, 7168


@triton.jit
def noop(X):
    pass


def cpu_time(fn, n=500, warm=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return (t1 - t0) / n * 1e6, (t2 - t0) / n * 1e6


x = torch.zeros(8, device=DEV)
print("triton noop        enq=%6.2f wall=%6.2f" % cpu_time(lambda: noop[(1,)](x)))

# pre-warmed kernel object, direct .run
print("triton noop .run   enq=%6.2f wall=%6.2f"
      % cpu_time(lambda: noop.run(x, grid=(1,), warmup=False)))

o = torch.empty((1, 256), device=DEV, dtype=torch.float16)
print("torch.empty        enq=%6.2f wall=%6.2f"
      % cpu_time(lambda: torch.empty((1, 256), device=DEV, dtype=torch.float16)))

A = torch.randn(64, K, device=DEV, dtype=torch.float16)
B = torch.randn(N, K, device=DEV, dtype=torch.float16)
print("B.T                enq=%6.2f wall=%6.2f" % cpu_time(lambda: B.T))
Bt = B.T
print("matmul(A,Bt)       enq=%6.2f wall=%6.2f" % cpu_time(lambda: torch.matmul(A, Bt)))
print("matmul(A,B.T)      enq=%6.2f wall=%6.2f" % cpu_time(lambda: torch.matmul(A, B.T)))
C = torch.empty((64, N), device=DEV, dtype=torch.float16)
print("mm out=            enq=%6.2f wall=%6.2f" % cpu_time(lambda: torch.mm(A, Bt, out=C)))

# What is the pure GPU time of matmul at small M? enqueue many then measure
for M in [1, 53, 901, 14104]:
    A = torch.randn(M, K, device=DEV, dtype=torch.float16)
    Bt = B.T
    for _ in range(20):
        torch.matmul(A, Bt)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    # Serialize enqueue first is impossible; instead use many streams? Just report event time
    # over 200 iters (enqueue-bound floor visible)
    s.record()
    for _ in range(200):
        torch.matmul(A, Bt)
    e.record()
    torch.cuda.synchronize()
    print(f"M={M:6d} event/iter={s.elapsed_time(e)/200*1e3:7.2f}us")
