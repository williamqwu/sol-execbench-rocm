import torch, time
import torch.nn.functional as F
def bench(f,n=60):
    for _ in range(20): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
C=512
for B,H,W in [(1,32,32),(16,32,32),(32,32,32),(2,64,64)]:
    S=H*W
    a=torch.randn(B,S,C,device='cuda')       # attention output (B,S,C)
    ar=torch.randn(B,C,H,W,device='cuda')
    ht=a.transpose(1,2).view(B,C,H,W)        # non-contiguous view
    ref=ht+ar
    t0=bench(lambda: a.transpose(1,2).view(B,C,H,W)+ar)
    t1=bench(lambda: a.transpose(1,2).contiguous().view(B,C,H,W)+ar)
    o1=a.transpose(1,2).contiguous().view(B,C,H,W)+ar
    print(f"B{B} {H}x{W}: strided_add={t0:.4f} contig+add={t1:.4f} mismatch={(o1!=ref).sum().item()}")
