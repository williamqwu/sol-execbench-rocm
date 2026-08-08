import torch, time
def bench(f,n=80):
    for _ in range(25): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
C=512
for B,H,W in [(1,32,32),(16,32,32),(32,32,32),(2,64,64)]:
    S=H*W
    a=torch.randn(B,S,C,device='cuda'); ar=torch.randn(B,C,H,W,device='cuda')
    strided=a.transpose(1,2).view(B,C,H,W)
    o=strided+ar
    print(f"B{B} {H}x{W}: (strided+ar).is_contiguous={o.is_contiguous()}")
    r=(strided+ar).contiguous()
    o2=torch.add(strided,ar,out=torch.empty_like(ar))
    t0=bench(lambda:(strided+ar).contiguous())
    t1=bench(lambda: torch.add(strided,ar,out=torch.empty_like(ar)))
    print(f"   contig={t0:.4f} out=({t1:.4f}) exact={(o2==r).all().item()}")
