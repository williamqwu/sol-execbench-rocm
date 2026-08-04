import torch, time
torch.cuda.init()
dev='cuda:0'

def bench(fn, iters=20, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t)/iters*1e3

# fp32 gemm peak probe
for M,K,N in [(8192,8192,8192),(131072,512,2048),(512,131072,2048),(131072,2048,512),(2048,131072,512)]:
    a=torch.randn(M,K,device=dev); b=torch.randn(K,N,device=dev)
    ms=bench(lambda: a@b)
    fl=2*M*K*N
    print(f"fp32 {M}x{K}x{N}: {ms:.3f} ms  {fl/ms*1e-9:.1f} TFLOP/s")

# bf16
for M,K,N in [(8192,8192,8192)]:
    a=torch.randn(M,K,device=dev,dtype=torch.bfloat16); b=torch.randn(K,N,device=dev,dtype=torch.bfloat16)
    ms=bench(lambda: a@b)
    print(f"bf16 {M}x{K}x{N}: {ms:.3f} ms  {2*M*K*N/ms*1e-9:.1f} TFLOP/s")

# tf32 probe
torch.backends.cuda.matmul.allow_tf32=True
a=torch.randn(8192,8192,device=dev); b=torch.randn(8192,8192,device=dev)
ms=bench(lambda: a@b)
print(f"tf32-allowed fp32 8192^3: {ms:.3f} ms  {2*8192**3/ms*1e-9:.1f} TFLOP/s")
torch.backends.cuda.matmul.allow_tf32=False

# bandwidth
x=torch.randn(1<<28,device=dev)
ms=bench(lambda: x.clone())
print(f"copy BW: {2*x.numel()*4/ms*1e-9:.1f} GB/s")
