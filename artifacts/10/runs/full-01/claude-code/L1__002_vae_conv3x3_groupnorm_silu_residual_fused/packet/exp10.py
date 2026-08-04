import torch, triton, triton.language as tl
dev='cuda:0'
torch.manual_seed(0)

# what's available
import triton.language.extra as ex
print('extra:', [n for n in dir(ex) if 'lib' in n.lower() or 'hip' in n.lower() or 'cuda' in n.lower()])
try:
    from triton.language.extra import libdevice
    print('libdevice exp?', hasattr(libdevice,'exp'), 'exp2?', hasattr(libdevice,'exp2'))
except Exception as e: print('no libdevice', e)

@triton.jit
def k(X, Y, N, V: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK); m=off<N
    x = tl.load(X+off, mask=m)
    if V==0:
        y = x/(1.0+tl.exp(-x))
    elif V==1:
        from triton.language.extra import libdevice
        y = x/(1.0+libdevice.exp(-x))
    elif V==2:
        # precise division too
        from triton.language.extra import libdevice
        e = libdevice.exp(-x)
        y = tl.fdiv(x, 1.0+e, ieee_rounding=True)
    elif V==3:
        y = tl.fdiv(x, 1.0+tl.exp(-x), ieee_rounding=True)
    tl.store(Y+off, y, mask=m)

import torch.nn.functional as F
N=1<<22
x=(torch.rand(N,device=dev)*40-20)
y=torch.empty_like(x)
ref=F.silu(x)
for v,name in [(0,'tl.exp'),(1,'libdevice.exp'),(2,'libdev+ieee_div'),(3,'tl.exp+ieee_div')]:
    try:
        k[(triton.cdiv(N,1024),)](x,y,N,v,1024)
        o=y.clone(); ne=(o!=ref).sum().item()
        print(f'  silu {name:20s} mismatch_frac={ne/N:.6f} maxabs={(o-ref).abs().max().item():.3e}')
    except Exception as e: print(' ',name,'ERR',repr(e)[:200])
