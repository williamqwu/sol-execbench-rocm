import torch, time
dev='cuda'
def bench(f,n=20):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000
print("default fp32 precision:", torch.backends.cuda.matmul.allow_tf32)
a=torch.randn(8192,8192,device=dev); b=torch.randn(8192,8192,device=dev)
t=bench(lambda:a@b); print("fp32 ieee mm TFLOPs:", 2*8192**3/t*1e-9)
torch.backends.cuda.matmul.allow_tf32=True
t=bench(lambda:a@b); print("tf32 mm TFLOPs:", 2*8192**3/t*1e-9)
r=(a@b); torch.backends.cuda.matmul.allow_tf32=False
r2=a@b
print("tf32 vs fp32 rel:", ((r-r2).abs().max()/r2.abs().std()).item())
