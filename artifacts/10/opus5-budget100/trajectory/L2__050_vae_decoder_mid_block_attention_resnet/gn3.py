import torch, time
import torch.nn.functional as F
def bench(f,n=100):
    for _ in range(30): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
C=512
for B,H,W in [(1,32,32),(1,16,16),(4,16,16),(2,41,41),(1,61,61),(1,48,48)]:
    S=H*W
    x=torch.randn(B,C,H,W,device='cuda')
    g=torch.randn(C,device='cuda'); b=torch.randn(C,device='cuda')
    r=torch.native_group_norm(x,g,b,B,C,S,32,1e-6)[0]
    # merge batch into groups: treat as (1, B*C, S) with B*32 groups -> same partition!
    xm=x.reshape(1,B*C,S)
    gm=g.repeat(B); bm=b.repeat(B)
    om=torch.native_group_norm(xm,gm,bm,1,B*C,S,B*32,1e-6)[0].view(B,C,H,W)
    t0=bench(lambda: torch.native_group_norm(x,g,b,B,C,S,32,1e-6))
    t1=bench(lambda: torch.native_group_norm(xm,gm,bm,1,B*C,S,B*32,1e-6))
    print(f"B{B} {H}x{W}: normal={t0:.4f} merged={t1:.4f} exact={(om==r).all().item()}")
