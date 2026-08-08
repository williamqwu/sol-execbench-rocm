import torch, time
def bench(f, n=50):
    for _ in range(10): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n
for M,K,N in [(8192,8192,8192),(4096,512,1536),(32768,512,512)]:
    a=torch.randn(M,K,device='cuda'); b=torch.randn(K,N,device='cuda')
    t=bench(lambda: a@b); print(f"fp32 {M}x{K}x{N}: {t*1e3:.3f}ms {2*M*K*N/t/1e12:.1f} TF")
a=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); b=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16)
t=bench(lambda: a@b); print(f"bf16 8192^3: {t*1e3:.3f}ms {2*8192**3/t/1e12:.1f} TF")
# bandwidth
x=torch.randn(1<<28,device='cuda'); y=torch.empty_like(x)
t=bench(lambda: y.copy_(x)); print(f"copy BW: {2*x.numel()*4/t/1e12:.2f} TB/s")
