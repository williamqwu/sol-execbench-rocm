import torch, time
import torch.nn.functional as F
def bench(f,n=30):
    for _ in range(8): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
w=torch.randn(512,512,3,3,device='cuda')/22; b=torch.randn(512,device='cuda')
for B,H,W in [(1,61,61),(1,48,48),(16,32,32),(1,16,16)]:
    x=torch.randn(B,512,H,W,device='cuda')
    torch.backends.cudnn.benchmark=False
    o0=F.conv2d(x,w,b,padding=1); t0=bench(lambda: F.conv2d(x,w,b,padding=1))
    torch.backends.cudnn.benchmark=True
    o1=F.conv2d(x,w,b,padding=1); t1=bench(lambda: F.conv2d(x,w,b,padding=1))
    torch.backends.cudnn.benchmark=False
    o2=F.conv2d(x,w,b,padding=1); t2=bench(lambda: F.conv2d(x,w,b,padding=1))
    print(f"B{B} {H}x{W}: off={t0:.4f} on={t1:.4f} back={t2:.4f} | on!=off:{(o1!=o0).sum().item()} back!=off:{(o2!=o0).sum().item()}")
