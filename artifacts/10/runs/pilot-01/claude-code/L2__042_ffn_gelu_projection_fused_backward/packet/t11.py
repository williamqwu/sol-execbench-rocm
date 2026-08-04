from lt import *
import torch, triton, triton.language as tl
from triton.language.extra import libdevice
SP=tl.constexpr(0.7978845608028654); C=tl.constexpr(0.044715)

# Pass torch-computed tanh in, do ONLY the arithmetic -> isolates FMA contraction
@triton.jit
def karith(X,T,GG,O,n,BLK: tl.constexpr):
    p=tl.program_id(0)*BLK+tl.arange(0,BLK); m=p<n
    x=tl.load(X+p,mask=m,other=0.); t=tl.load(T+p,mask=m,other=0.); gg=tl.load(GG+p,mask=m,other=0.)
    xc = x*x*x
    dt = SP*(1.0 + (3.0*C)*(x*x))
    sq = 1.0 - t*t
    gr = 0.5*(1.0+t) + 0.5*x*sq*dt
    tl.store(O+p, gg*gr, mask=m)

torch.manual_seed(0)
x=(torch.randn(1<<22,device=dev)*2).contiguous(); gg=torch.randn_like(x)
sp=0.7978845608028654; c=0.044715
t_=torch.tanh(sp*(x+c*(x*x*x)))
dt_=sp*(1.0+3.0*c*x*x); sq_=1.0-t_*t_
ref=gg*(0.5*(1.0+t_)+0.5*x*sq_*dt_)
o=torch.empty_like(x); karith[(triton.cdiv(x.numel(),1024),)](x,t_,gg,o,x.numel(),1024)
print("arith-only bitexact:",torch.equal(o,ref),"fracdiff",(o!=ref).float().mean().item())

# check torch's own eval order for dtanh: 3.0*coeff*x*x  -> ((3.0*c)*x)*x  vs 3*c*(x*x)?
a1=3.0*c*x*x   # python: ((3.0*c)*x)*x
a2=(3.0*c)*(x*x)
print("dtanh assoc same:",torch.equal(a1,a2))
b1=0.5*x*sq_*dt_   # ((0.5*x)*sq)*dt
b2=0.5*(x*sq_*dt_)
print("term assoc same:",torch.equal(b1,b2))
