import torch, time
import torch.nn.functional as F
def bench(f,n=60):
    for _ in range(20): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
C=512
for B,H,W in [(16,32,32),(32,32,32)]:
    x=torch.randn(B,C,H,W,device='cuda'); tp=torch.randn(B,C,device='cuda')
    r = x + tp[:,:,None,None]
    # alternatives
    o1 = x.add(tp.view(B,C,1,1))
    o2 = torch.add(x, tp.view(B,C,1,1))
    xv = x.view(B,C,H*W)
    o3 = (xv + tp.unsqueeze(-1)).view(B,C,H,W)
    t0=bench(lambda: x+tp[:,:,None,None])
    t3=bench(lambda: (xv+tp.unsqueeze(-1)))
    ti=bench(lambda: x.clone().add_(tp.view(B,C,1,1)))
    print(f"B{B} {H}x{W}: bcast={t0:.4f} 3d={t3:.4f} inplace+clone={ti:.4f} | m3={(o3!=r).sum().item()}")
    # residual add: can we do it in-place on h (h is freshly produced by conv, safe to mutate)?
    y=torch.randn_like(x)
    ta=bench(lambda: x+y); tia=bench(lambda: x.clone().add_(y))
    print(f"   add={ta:.4f} clone+add_={tia:.4f}")
