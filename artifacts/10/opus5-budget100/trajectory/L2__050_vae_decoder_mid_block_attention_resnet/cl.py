import torch, time
import torch.nn.functional as F
def bench(f,n=40):
    for _ in range(12): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
w=torch.randn(512,512,3,3,device='cuda')/22; b=torch.randn(512,device='cuda')
wcl=w.to(memory_format=torch.channels_last)
for B,H,W in [(1,32,32),(16,32,32),(32,32,32),(2,64,64),(4,48,48),(1,16,16)]:
    x=torch.randn(B,512,H,W,device='cuda')
    xcl=x.to(memory_format=torch.channels_last)
    r=F.conv2d(x,w,b,padding=1)
    o=F.conv2d(xcl,wcl,b,padding=1)
    t0=bench(lambda: F.conv2d(x,w,b,padding=1))
    t1=bench(lambda: F.conv2d(xcl,wcl,b,padding=1))
    tconv=bench(lambda: x.to(memory_format=torch.channels_last))
    print(f"B{B} {H}x{W}: nchw={t0:.4f} nhwc={t1:.4f} cvt={tconv:.4f} | mismatch={(o.contiguous()!=r).sum().item()} max={(o.contiguous()-r).abs().max().item():.2e}")
