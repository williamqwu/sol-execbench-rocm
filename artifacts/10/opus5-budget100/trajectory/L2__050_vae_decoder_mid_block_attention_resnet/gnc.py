import torch, time
import torch.nn.functional as F
def bench(f,n=60):
    for _ in range(20): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
C=512
g=torch.ones(C,device='cuda'); b=torch.zeros(C,device='cuda')
for B,H,W in [(1,32,32),(16,32,32),(32,32,32),(1,16,16),(2,64,64)]:
    x=torch.randn(B,C,H,W,device='cuda')
    S=H*W
    x3=x.view(B,C,S)
    r=F.group_norm(x,32,g,b,1e-6)
    o3=F.group_norm(x3,32,g,b,1e-6).view(B,C,H,W)
    o2=F.group_norm(x.view(B,C,S,1),32,g,b,1e-6).view(B,C,H,W)
    t0=bench(lambda: F.group_norm(x,32,g,b,1e-6))
    t3=bench(lambda: F.group_norm(x3,32,g,b,1e-6))
    # native_group_norm direct
    tn=bench(lambda: torch.native_group_norm(x,g,b,B,C,S,32,1e-6))
    print(f"B{B} {H}x{W}: 4d={t0:.4f} 3d={t3:.4f} native={tn:.4f} | m3={(o3!=r).sum().item()} m2={(o2!=r).sum().item()}")
