import torch, triton, triton.language as tl
@triton.jit
def k(a_ptr,s_ptr,o1,o2,N):
    i=tl.arange(0,1024)
    m=i<N
    a=tl.load(a_ptr+i,mask=m,other=1.0)
    s=tl.load(s_ptr+i,mask=m,other=1.0)
    tl.store(o1+i, a/s, mask=m)
    tl.store(o2+i, tl.fdiv(a,s,ieee_rounding=True), mask=m)
torch.manual_seed(0)
N=1024
a=(torch.rand(N,device='cuda')*100).float()
s=(torch.rand(N,device='cuda')+0.1).float()
o=[torch.empty(N,device='cuda') for _ in range(2)]
k[(1,)](a,s,*o,N)
ref=a/s
for nm,x in zip(["tt_div","tt_fdiv_ieee"],o):
    print(nm,"exact:",torch.equal(x,ref),"ndiff:",(x!=ref).sum().item())
# scalar-div lowering check
amax=(torch.rand(N,device='cuda')*10).float()
print("scalar div == recip-mul:", torch.equal(amax/448.0, amax*(1.0/448.0)))
print("scalar div == true div :", torch.equal(amax/448.0, torch.full_like(amax,448.0).reciprocal()*0+torch.div(amax, torch.full_like(amax,448.0))))
