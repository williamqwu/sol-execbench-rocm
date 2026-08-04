import torch, time, torch.nn.functional as F
torch.manual_seed(0)
dev='cuda'
def bench(f,n=20):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000

for (B,H,W) in [(32,32,32),(1,32,32),(2,64,64),(4,16,16)]:
    C=512
    x=torch.randn(B,C,H,W,device=dev); w=torch.randn(C,C,3,3,device=dev); b=torch.randn(C,device=dev)
    ref=F.conv2d(x,w,b,padding=1)
    t0=bench(lambda: F.conv2d(x,w,b,padding=1))
    torch.backends.cudnn.benchmark=True
    for _ in range(3): F.conv2d(x,w,b,padding=1)
    t1=bench(lambda: F.conv2d(x,w,b,padding=1))
    torch.backends.cudnn.benchmark=False
    wf=w.reshape(C,C*9)
    def im2col():
        col=F.unfold(x,3,padding=1)   # [B, C*9, H*W]
        return (wf@col).add_(b[:,None]).view(B,C,H,W)
    o=im2col()
    t2=bench(im2col)
    err=(o-ref).abs().max().item()/ref.abs().max().item()
    print(f"B{B} {H}x{W}: conv={t0:.3f} bench={t1:.3f} im2col={t2:.3f} relerr={err:.2e}")
