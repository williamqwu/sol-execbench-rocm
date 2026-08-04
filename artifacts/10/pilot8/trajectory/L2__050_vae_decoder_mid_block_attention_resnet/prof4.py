import torch, time
dev='cuda'
def bench(f,n=20):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000
M,K,N=512,4608,32768
a=torch.randn(M,K,device=dev); b=torch.randn(K,N,device=dev)
t=bench(lambda:a@b); print("gemm 512x4608x32768:",t,"TFLOPS",2*M*K*N/t*1e-9)
# accuracy check fp32
a2=torch.randn(1024,1024,device=dev); b2=torch.randn(1024,1024,device=dev)
r=(a2@b2).double(); r64=(a2.double()@b2.double())
print("fp32 mm rel err:", ((r-r64).abs().max()/r64.abs().max()).item())
# nt layout
bt=b.t().contiguous()
print("gemm NT:",bench(lambda:a@bt.t()))
M2,K2,N2=32768,4608,512
a3=torch.randn(M2,K2,device=dev); w3=torch.randn(N2,K2,device=dev)
print("gemm 32768x4608x512 NT:",bench(lambda: a3@w3.t()))
