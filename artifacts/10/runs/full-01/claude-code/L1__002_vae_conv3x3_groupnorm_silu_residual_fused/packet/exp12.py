import torch, triton, triton.language as tl, torch.nn.functional as F, numpy as np
from triton.language.extra import libdevice
dev='cuda:0'
torch.manual_seed(0)
N=1<<22
x=(torch.rand(N,device=dev)*40-20)

# Is torch.exp correctly rounded (== fp64 exp rounded to fp32)?
xr = x.double()
e64 = torch.exp(xr).float()
et = torch.exp(x)
print('torch.exp vs fp64-rounded: mismatch=%.6f'%((e64!=et).float().mean().item()))

# torch silu vs fp64 formula rounded
s64 = (xr/(1+torch.exp(-xr))).float()
st = F.silu(x)
print('torch.silu vs fp64 x/(1+exp(-x)) rounded: mismatch=%.6f'%((s64!=st).float().mean().item()))
# torch silu vs fp32 x/(1+torch.exp(-x))
s32 = x/(1+torch.exp(-x))
print('torch.silu vs fp32 x/(1+torch.exp(-x)): mismatch=%.6f'%((s32!=st).float().mean().item()))

@triton.jit
def k(X, Y, N, V: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK); m=off<N
    x = tl.load(X+off, mask=m)
    if V==0:
        xd = x.to(tl.float64)
        y = (xd/(1.0+tl.exp(-xd))).to(tl.float32)
    elif V==1:
        xd = x.to(tl.float64)
        y = (xd/(1.0+libdevice.exp(-xd))).to(tl.float32)
    elif V==2:
        xd = x.to(tl.float64)
        e = libdevice.exp(-xd)
        y = tl.fdiv(xd, 1.0+e, ieee_rounding=True).to(tl.float32)
    elif V==3:
        e = libdevice.exp(-x.to(tl.float64)).to(tl.float32)
        y = x/(1.0+e)
    tl.store(Y+off, y, mask=m)

y=torch.empty_like(x)
for v,name in [(0,'fp64 tl.exp'),(1,'fp64 libdev.exp'),(2,'fp64 libdev ieee'),(3,'fp64exp->fp32 div')]:
    try:
        k[(triton.cdiv(N,1024),)](x,y,N,v,1024); o=y.clone()
        print(f'  silu {name:20s} mismatch={(o!=st).float().mean().item():.6f} maxabs={(o-st).abs().max().item():.3e}')
    except Exception as e: print(' ',name,'ERR',repr(e)[:200])
