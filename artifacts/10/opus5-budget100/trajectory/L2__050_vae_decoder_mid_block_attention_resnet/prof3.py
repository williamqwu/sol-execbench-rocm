import torch, time
import torch.nn.functional as F
dev='cuda'
def bench(fn, iters=30, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1000

# GEMM throughput fp32
for (M,N,K) in [(32768,512,4608),(1024,512,4608),(8192,512,4608),(32768,512,512)]:
    a=torch.randn(M,K,device=dev); b=torch.randn(K,N,device=dev)
    t=bench(lambda: a@b)
    fl=2*M*N*K
    print(f"fp32 gemm {M}x{N}x{K}: {t:.3f} ms  {fl/t*1e-9:.1f} TFLOP/s")
    ab=a.bfloat16(); bb=b.bfloat16()
    t=bench(lambda: ab@bb)
    print(f"  bf16: {t:.3f} ms  {fl/t*1e-9:.1f} TFLOP/s")
