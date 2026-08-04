from lt import *
import torch, triton, triton.language as tl
from triton.language.extra import libdevice
SP=tl.constexpr(0.7978845608028654); C=tl.constexpr(0.044715); C3=tl.constexpr(3.0*0.044715)

@triton.jit
def kfull(X,GG,O,n,BLK: tl.constexpr):
    p=tl.program_id(0)*BLK+tl.arange(0,BLK); m=p<n
    x=tl.load(X+p,mask=m,other=0.); gg=tl.load(GG+p,mask=m,other=0.)
    t = libdevice.tanh(SP*(x + C*(x*x*x)))
    dt = SP*(1.0 + (C3*x)*x)
    sq = 1.0 - t*t
    tl.store(O+p, gg*(0.5*(1.0+t) + ((0.5*x)*sq)*dt), mask=m)

torch.manual_seed(0)
x=(torch.randn(1<<22,device=dev)*2).contiguous(); gg=torch.randn_like(x)
sp=0.7978845608028654; c=0.044715
t_=torch.tanh(sp*(x+c*(x*x*x)))
ref=gg*(0.5*(1.0+t_)+0.5*x*(1.0-t_*t_)*(sp*(1.0+3.0*c*x*x)))
for fp in [True,False]:
    o=torch.empty_like(x); kfull[(triton.cdiv(x.numel(),1024),)](x,gg,o,x.numel(),1024,enable_fp_fusion=fp)
    print(f"fp_fusion={fp}: bitexact={torch.equal(o,ref)} fracdiff={(o!=ref).float().mean().item():.5f} maxerr={(o-ref).abs().max().item():.3e}")
