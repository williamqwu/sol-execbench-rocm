import torch, time
import torch.nn.functional as F
def bench(f,n=50):
    for _ in range(15): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
scale = 512**-0.5
for B,S in [(2,4096),(4,2304),(1,3721),(16,1024),(32,1024)]:
    q=torch.randn(B,S,512,device='cuda'); k=torch.randn(B,S,512,device='cuda')
    ref = torch.matmul(q,k.transpose(-2,-1))*scale
    z = torch.zeros(1, device='cuda')
    o1 = torch.baddbmm(z, q, k.transpose(-2,-1), beta=0, alpha=scale)
    t0=bench(lambda: torch.matmul(q,k.transpose(-2,-1))*scale)
    t1=bench(lambda: torch.baddbmm(z, q, k.transpose(-2,-1), beta=0, alpha=scale))
    print(f"B{B} S{S}: sep={t0:.4f} baddbmm={t1:.4f} mismatch={(o1!=ref).sum().item()}")
