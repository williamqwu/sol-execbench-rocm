import torch, time
import torch.nn.functional as F
def bench(f,n=50):
    for _ in range(15): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
for B,S in [(2,4096),(4,2304),(1,3721),(16,1024),(32,1024)]:
    s=torch.randn(B,S,S,device='cuda')
    v=torch.randn(B,S,512,device='cuda')
    t_sm=bench(lambda: F.softmax(s,dim=-1))
    p=F.softmax(s,dim=-1)
    t_pv=bench(lambda: torch.matmul(p,v))
    # in-place softmax?
    sc=s.clone()
    ref=F.softmax(s,dim=-1)
    o=torch.softmax(sc,dim=-1,out=sc) if False else None
    print(f"B{B} S{S}: softmax={t_sm:.4f} pv={t_pv:.4f}  smBW={3*B*S*S*4/(t_sm*1e-3)/1e12:.2f}TB/s  pvTF={2*B*S*S*512/(t_pv*1e-3)/1e12:.1f}")
