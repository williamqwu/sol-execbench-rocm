import torch, triton, triton.language as tl
dev='cuda:0'
# Q1: does torch scalar-divide == recip-mul?
torch.manual_seed(0)
x=torch.rand(1<<20,device=dev)*100
print("torch x/448.0 == x*(1/448):", torch.equal(x/448.0, x*(1.0/448.0)))
print("torch x/448.0 == x/tensor(448):", torch.equal(x/448.0, x/torch.tensor(448.0,device=dev)))

# Q2: tl.fdiv ieee_rounding for tensor/tensor
@triton.jit
def k(A,B,O1,O2,O3,BLK: tl.constexpr):
    o=tl.arange(0,BLK)
    a=tl.load(A+o); b=tl.load(B+o)
    tl.store(O1+o, a/b)
    tl.store(O2+o, tl.fdiv(a,b,ieee_rounding=True))
    tl.store(O3+o, tl.fdiv(a,b,ieee_rounding=False))
n=1<<20
a=(torch.rand(n,device=dev)*2-1)*100
b=torch.rand(n,device=dev)*0.05+1e-4
o1=torch.empty(n,device=dev);o2=torch.empty(n,device=dev);o3=torch.empty(n,device=dev)
k[(triton.cdiv(n,1024),)](a,b,o1,o2,o3,BLK=1024) if False else None
@triton.jit
def k2(A,B,O1,O2,O3,n,BLK: tl.constexpr):
    o=tl.program_id(0)*BLK+tl.arange(0,BLK); m=o<n
    x=tl.load(A+o,mask=m); y=tl.load(B+o,mask=m,other=1.)
    tl.store(O1+o, x/y, mask=m)
    tl.store(O2+o, tl.fdiv(x,y,ieee_rounding=True), mask=m)
    tl.store(O3+o, tl.fdiv(x,y,ieee_rounding=False), mask=m)
k2[(triton.cdiv(n,1024),)](a,b,o1,o2,o3,n,BLK=1024)
ref=a/b
def bit(p,q): return (p.view(torch.int32)==q.view(torch.int32)).float().mean().item()
print("plain /        vs torch:", bit(o1,ref))
print("fdiv ieee=True vs torch:", bit(o2,ref))
print("fdiv ieee=False vs torch:", bit(o3,ref))
