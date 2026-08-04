from lt import *
import torch, triton, triton.language as tl
from triton.language.extra import libdevice
SP=tl.constexpr(0.7978845608028654); C=tl.constexpr(0.044715); C3=tl.constexpr(3.0*0.044715)

@triton.jit
def kv(X,GG,O,n,VAR: tl.constexpr,BLK: tl.constexpr):
    p=tl.program_id(0)*BLK+tl.arange(0,BLK); m=p<n
    x=tl.load(X+p,mask=m,other=0.); gg=tl.load(GG+p,mask=m,other=0.)
    if VAR==0:
        t=libdevice.tanh(SP*(x+C*(x*x*x)))
        d=SP*(1.0+C3*(x*x))
    elif VAR==1:
        t=libdevice.tanh(SP*(x+C*(x*x*x)))
        d=SP*(1.0+(C3*x)*x)
    elif VAR==2:
        t=libdevice.tanh(SP*(x+C*(x*x*x)))
        d=SP*(1.0+(C3*x)*x)
    o=gg*(0.5*(1.0+t)+((0.5*x)*(1.0-t*t))*d)
    tl.store(O+p,o,mask=m)

torch.manual_seed(0)
x=(torch.randn(1<<22,device=dev)*2).contiguous()
gg=torch.randn_like(x)
sp=0.7978845608028654; c=0.044715
xc=x*x*x
t_=torch.tanh(sp*(x+c*xc))
d_=sp*(1.0+3.0*c*x*x)
s_=1.0-t_*t_
ref=gg*(0.5*(1.0+t_)+0.5*x*s_*d_)
for v in [0,1,2]:
    o=torch.empty_like(x); kv[(triton.cdiv(x.numel(),1024),)](x,gg,o,x.numel(),v,1024)
    ne=(o!=ref).float().mean().item()
    print(f"var{v}: bitexact={torch.equal(o,ref)} frac_diff={ne:.4f} maxulpdiff={(o-ref).abs().max().item():.3e}")
# tanh alone
@triton.jit
def kt(X,O,n,BLK: tl.constexpr):
    p=tl.program_id(0)*BLK+tl.arange(0,BLK); m=p<n
    tl.store(O+p, libdevice.tanh(tl.load(X+p,mask=m,other=0.)), mask=m)
a=sp*(x+c*xc)
o=torch.empty_like(x); kt[(triton.cdiv(x.numel(),1024),)](a,o,x.numel(),1024)
print("tanh alone bitexact", torch.equal(o,t_), "fracdiff",(o!=t_).float().mean().item())
