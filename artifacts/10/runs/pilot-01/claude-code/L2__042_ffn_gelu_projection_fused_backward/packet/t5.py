from lt import *
import torch, triton, triton.language as tl
from triton.language.extra import libdevice
B,S=32,4096; H=512; I=2048; BS=B*S
inp=gen(B,S)
f1o=inp['fc1_output'].view(BS,I)
# tanh bit-exactness
@triton.jit
def k_tanh(X, O, n, BLK: tl.constexpr):
    p=tl.program_id(0)*BLK+tl.arange(0,BLK)
    m=p<n
    x=tl.load(X+p,mask=m,other=0.)
    tl.store(O+p, libdevice.tanh(x), mask=m)
x=f1o.reshape(-1)[:1<<22].contiguous()
o=torch.empty_like(x)
k_tanh[(triton.cdiv(x.numel(),1024),)](x,o,x.numel(),1024)
r=torch.tanh(x)
print("tanh bitexact:", torch.equal(o,r), "maxdiff", (o-r).abs().max().item())
# also test tanh on wide range
y=(torch.randn(1<<20,device=dev)*5)
o=torch.empty_like(y); k_tanh[(triton.cdiv(y.numel(),1024),)](y,o,y.numel(),1024)
print("tanh2 bitexact:", torch.equal(o,torch.tanh(y)))
