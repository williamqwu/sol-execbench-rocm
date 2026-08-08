import torch, time, os
import torch.nn.functional as F
dev='cuda'
def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1000

C=512
print("tf32 avail:", torch.backends.cuda.matmul.allow_tf32)
cases=[(1,32,32),(32,32,32),(2,64,64),(1,61,61),(4,16,16)]
w = torch.randn(C,C,3,3,device=dev)*0.01
b = torch.randn(C,device=dev)

for bench_mode in [False, True]:
    torch.backends.cudnn.benchmark = bench_mode
    for (B,H,W) in cases:
        x = torch.randn(B,C,H,W,device=dev)
        t = bench(lambda: F.conv2d(x,w,b,padding=1))
        xc = x.to(memory_format=torch.channels_last); wc=w.to(memory_format=torch.channels_last)
        tc = bench(lambda: F.conv2d(xc,wc,b,padding=1))
        print(f"benchmark={bench_mode} B{B} {H}x{W}: nchw={t:.3f} nhwc={tc:.3f}")
