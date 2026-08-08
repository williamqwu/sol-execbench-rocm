import torch, time
import torch.nn.functional as F
def bench(f,n=80):
    for _ in range(25): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
C=512
for B,H,W in [(1,32,32),(16,32,32),(32,32,32),(2,64,64),(1,16,16)]:
    x=torch.randn(B,C,H,W,device='cuda'); tp=torch.randn(B,C,device='cuda'); y=torch.randn_like(x)
    tpv=tp.view(B,C,1,1)
    r1=x+tpv; r2=F.silu(x); r3=x+y
    a=x.clone(); torch.add(a,tpv,out=a); print("  add_ exact:",(a==r1).all().item(), end='')
    b=x.clone(); F.silu(b,inplace=True); print(" silu_ exact:",(b==r2).all().item(), end='')
    c=x.clone(); c.add_(y); print(" res_ exact:",(c==r3).all().item())
    t0=bench(lambda: x+tpv); t1=bench(lambda: x.add_(tpv))
    s0=bench(lambda: F.silu(x)); s1=bench(lambda: F.silu(x,inplace=True))
    r0=bench(lambda: x+y); ri=bench(lambda: x.add_(y))
    print(f"B{B} {H}x{W}: bcast {t0:.4f}->{t1:.4f}  silu {s0:.4f}->{s1:.4f}  res {r0:.4f}->{ri:.4f}")
