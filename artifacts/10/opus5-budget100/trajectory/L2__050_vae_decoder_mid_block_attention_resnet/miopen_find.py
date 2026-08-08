import torch, os, sys, time
import torch.nn.functional as F
dev='cuda'
C=512
def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1000

torch.manual_seed(0)
w=(torch.randn(C,C,3,3,device=dev)/24); b=torch.randn(C,device=dev)
mode = os.environ.get('MIOPEN_FIND_ENFORCE','default')
bm = os.environ.get('BM','0')=='1'
torch.backends.cudnn.benchmark = bm
print(f"MIOPEN_FIND_ENFORCE={mode} benchmark={bm} FIND_MODE={os.environ.get('MIOPEN_FIND_MODE','-')}")
import hashlib
for (B,H,W) in [(1,32,32),(32,32,32),(2,64,64),(1,16,16),(4,48,48),(1,61,61)]:
    x=torch.randn(B,C,H,W,device=dev)
    o=F.conv2d(x,w,b,padding=1)
    t=bench(lambda: F.conv2d(x,w,b,padding=1))
    hsh = hashlib.md5(o.cpu().numpy().tobytes()).hexdigest()[:12]
    print(f"  B{B} {H}x{W}: {t:.4f} ms hash={hsh}")
