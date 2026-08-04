import torch, triton, triton.language as tl
E=tl.constexpr(448.0)
@triton.jit
def k(a_ptr,o1,o2,o3,N):
    i=tl.arange(0,1024)
    m=i<N
    a=tl.load(a_ptr+i,mask=m,other=1.0)
    tl.store(o1+i, a/E, mask=m)
    tl.store(o2+i, tl.fdiv(a,E,ieee_rounding=True), mask=m)
    tl.store(o3+i, a*(1.0/448.0), mask=m)
torch.manual_seed(0)
N=1024
a=(torch.rand(N,device='cuda')*100).float()
o=[torch.empty(N,device='cuda') for _ in range(3)]
k[(1,)](a,*o,N)
ref=a/448.0
for nm,x in zip(["div","fdiv_ieee","recip_mul"],o):
    print(nm,"exact:",torch.equal(x,ref),"ndiff:",(x!=ref).sum().item())
