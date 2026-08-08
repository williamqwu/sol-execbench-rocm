import torch, time
import torch.nn.functional as F
def bench(f,n=60):
    for _ in range(20): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
g=torch.ones(512,device='cuda'); b=torch.zeros(512,device='cuda')
for B,H,W in [(1,32,32),(16,32,32),(32,32,32),(2,64,64),(1,16,16)]:
    x=torch.randn(B,512,H,W,device='cuda'); nb=x.numel()*4
    t_gn=bench(lambda: F.group_norm(x,32,g,b,1e-6))
    t_si=bench(lambda: F.silu(x))
    t_ad=bench(lambda: x+x)
    print(f"B{B} {H}x{W} ({nb/1e6:.1f}MB): gn={t_gn:.4f}({3*nb/(t_gn*1e-3)/1e12:.2f}TB/s) silu={t_si:.4f}({2*nb/(t_si*1e-3)/1e12:.2f}) add={t_ad:.4f}({3*nb/(t_ad*1e-3)/1e12:.2f})")
