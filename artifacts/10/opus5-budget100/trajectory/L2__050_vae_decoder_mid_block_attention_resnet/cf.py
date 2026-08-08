import torch, time
import torch.nn.functional as F
def bench(f,n=50):
    for _ in range(15): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
C=512; scale=C**-0.5
for B,H,W in [(1,32,32),(2,64,64),(16,32,32),(32,32,32),(4,48,48)]:
    S=H*W
    gn=torch.randn(B,C,S,device='cuda')
    Wq=torch.randn(3*C,C,device='cuda')/22; bq=torch.randn(3*C,device='cuda')
    Wo=torch.randn(C,C,device='cuda')/22; bo=torch.randn(C,device='cuda')
    z=torch.zeros(1,device='cuda')
    def ref():
        h=gn.transpose(1,2)
        qkv=F.linear(h,Wq,bq); q,k,v=qkv.split(C,dim=-1)
        s=torch.baddbmm(z,q,k.transpose(-2,-1),beta=0.,alpha=scale)
        p=F.softmax(s,dim=-1)
        o=F.linear(torch.matmul(p,v),Wo,bo)
        return o.transpose(1,2).contiguous()
    def cf():
        # qkv_cs (B,3C,S) = Wq @ gn + bq
        qkv=torch.baddbmm(bq.view(1,3*C,1), Wq.unsqueeze(0).expand(B,3*C,C), gn)
        q,k,v=qkv.split(C,dim=1)
        s=torch.baddbmm(z,q.transpose(1,2),k,beta=0.,alpha=scale)
        p=F.softmax(s,dim=-1)
        pv=torch.bmm(v,p.transpose(1,2))   # (B,C,S)
        return torch.baddbmm(bo.view(1,C,1), Wo.unsqueeze(0).expand(B,C,C), pv)
    r=ref(); c=cf()
    t0=bench(ref); t1=bench(cf)
    print(f"B{B} {H}x{W}: ref={t0:.4f} cf={t1:.4f}  mismatch={(c!=r).sum().item()} max={(c-r).abs().max().item():.2e}")
