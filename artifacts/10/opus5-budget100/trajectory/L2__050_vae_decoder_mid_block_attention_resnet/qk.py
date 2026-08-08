import torch, time
import torch.nn.functional as F
def bench(f,n=50):
    for _ in range(15): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
C=512; scale=C**-0.5; z=torch.zeros(1,device='cuda')
for B,S in [(32,1024),(16,1024),(2,4096),(4,2304)]:
    q=torch.randn(B,S,C,device='cuda'); k=torch.randn(B,S,C,device='cuda')
    ref=torch.baddbmm(z,q,k.transpose(-2,-1),beta=0.,alpha=scale)
    kc=k.transpose(-2,-1).contiguous()   # (B,C,S) real
    o2=torch.baddbmm(z,q,kc,beta=0.,alpha=scale)
    t0=bench(lambda: torch.baddbmm(z,q,k.transpose(-2,-1),beta=0.,alpha=scale))
    t1=bench(lambda: torch.baddbmm(z,q,kc,beta=0.,alpha=scale))
    tc=bench(lambda: k.transpose(-2,-1).contiguous())
    F_=2*B*S*S*C
    print(f"B{B} S{S}: NT={t0:.4f}({F_/(t0*1e-3)/1e12:.1f}TF) NN={t1:.4f}({F_/(t1*1e-3)/1e12:.1f}TF) transp={tc:.4f} | mismatch={(o2!=ref).sum().item()}")
