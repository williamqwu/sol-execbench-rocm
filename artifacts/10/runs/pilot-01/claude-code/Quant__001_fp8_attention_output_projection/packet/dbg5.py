import torch, triton, triton.language as tl
dev='cuda:0'
@triton.jit
def k(X, O1, O2, O3, n, C, BLK: tl.constexpr):
    off=tl.arange(0,BLK)
    x=tl.load(X+off)
    tl.store(O1+off, x/448.0)                 # constexpr literal divide
    tl.store(O2+off, x/C)                     # runtime scalar divide
    tl.store(O3+off, x*(1.0/448.0))           # explicit reciprocal multiply
torch.manual_seed(0)
n=1<<14
x=torch.rand(n,device=dev)*100
o1=torch.empty(n,device=dev);o2=torch.empty(n,device=dev);o3=torch.empty(n,device=dev)
k[(1,)](x,o1,o2,o3,n,448.0,BLK=n)
ref=x/448.0
def bit(a,b): return (a.view(torch.int32)==b.view(torch.int32)).float().mean().item()
print("literal-div  vs torch:", bit(o1,ref))
print("runtime-div  vs torch:", bit(o2,ref))
print("recip-mul    vs torch:", bit(o3,ref))
print("literal vs recip-mul :", bit(o1,o3))
