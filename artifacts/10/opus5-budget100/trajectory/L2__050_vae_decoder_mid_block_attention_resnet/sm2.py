import torch, time
import torch.nn.functional as F
def bench(f,n=60):
    for _ in range(20): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
C=512; scale=C**-0.5; z=torch.zeros(1,device='cuda')
for B,S in [(2,4096),(16,1024),(32,1024),(1,3721)]:
    q=torch.randn(B,S,C,device='cuda'); k=torch.randn(B,S,C,device='cuda')
    s=torch.baddbmm(z,q,k.transpose(-2,-1),beta=0.,alpha=scale)
    r=F.softmax(s,dim=-1)
    sc=s.clone(); o=torch.softmax(sc,dim=-1,out=sc)
    print(f"B{B} S{S}: inplace softmax exact={(o==r).all().item()} oop={bench(lambda: F.softmax(s,dim=-1)):.4f} ip={bench(lambda: torch.softmax(s,dim=-1,out=s.clone())):.4f}")
    # softmax into pre-existing buffer to reuse s memory
    t_ip=bench(lambda: torch.softmax(s,dim=-1,out=s))
    print(f"   true-inplace(reuse s)={t_ip:.4f}")
