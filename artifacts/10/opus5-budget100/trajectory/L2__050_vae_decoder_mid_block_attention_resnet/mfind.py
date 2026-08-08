import os, sys
mode = sys.argv[1]
if mode == 'find': os.environ['MIOPEN_FIND_ENFORCE']='3'; os.environ['MIOPEN_FIND_MODE']='1'
import torch, time
import torch.nn.functional as F
if mode=='bench': torch.backends.cudnn.benchmark=True
def bench(f,n=30):
    for _ in range(8): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
w=torch.randn(512,512,3,3,device='cuda')/22; b=torch.randn(512,device='cuda')
res={}
for B,H,W in [(1,32,32),(1,61,61),(2,64,64),(1,48,48),(16,32,32),(4,16,16),(8,32,32),(32,32,32),(4,48,48),(2,41,41),(1,16,16)]:
    x=torch.randn(B,512,H,W,device='cuda')
    t=bench(lambda: F.conv2d(x,w,b,padding=1))
    o=F.conv2d(x,w,b,padding=1)
    print(f"{mode} B{B} {H}x{W}: {t:.4f}ms  cksum={o.double().sum().item():.10f}")
