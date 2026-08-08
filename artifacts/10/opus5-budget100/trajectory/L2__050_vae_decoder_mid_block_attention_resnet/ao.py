import torch, time
import torch.nn.functional as F
def bench(f,n=50):
    for _ in range(15): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
C=512
for B,S in [(2,4096),(4,2304),(16,1024),(32,1024),(1,256)]:
    pv=torch.randn(B,S,C,device='cuda'); w=torch.randn(C,C,device='cuda')/22; b=torch.randn(C,device='cuda')
    ref=F.linear(pv,w,b)
    # baddbmm form with broadcast bias
    wt = w.t().contiguous().unsqueeze(0).expand(B,C,C)
    o2 = torch.baddbmm(b.view(1,1,C), pv, wt)
    t0=bench(lambda: F.linear(pv,w,b))
    t1=bench(lambda: torch.baddbmm(b.view(1,1,C), pv, wt))
    print(f"B{B} S{S}: linear={t0:.4f} baddbmm={t1:.4f} mismatch={(o2!=ref).sum().item()}")
