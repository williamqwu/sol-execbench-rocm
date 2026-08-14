import torch, triton, triton.language as tl, torch.nn.functional as F
from triton.language.extra import libdevice
dev='cuda:0'
torch.manual_seed(0)

@triton.jit
def k(X, Y, N, V: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK); m=off<N
    x = tl.load(X+off, mask=m)
    if V==0:   y = x/(1.0+tl.exp(-x))
    elif V==1: y = x/(1.0+libdevice.exp(-x))
    elif V==2: y = tl.fdiv(x, 1.0+libdevice.exp(-x), ieee_rounding=True)
    elif V==3: y = x/(1.0+libdevice.exp2(-x*1.4426950408889634))
    elif V==4: y = x*libdevice.rcp64h(1.0+libdevice.exp(-x))
    elif V==5:
        e = libdevice.exp(-x); d = 1.0+e
        y = tl.fdiv(x, d, ieee_rounding=True)
    tl.store(Y+off, y, mask=m)

N=1<<22
x=(torch.rand(N,device=dev)*40-20); y=torch.empty_like(x); ref=F.silu(x)
for v,name in [(0,'tl.exp'),(1,'libdevice.exp'),(2,'libdev+ieeediv'),(3,'libdev.exp2'),(5,'libdev+ieeediv2')]:
    try:
        k[(triton.cdiv(N,1024),)](x,y,N,v,1024); o=y.clone()
        print(f'  {name:18s} mismatch={(o!=ref).sum().item()/N:.6f} maxabs={(o-ref).abs().max().item():.3e}')
    except Exception as e: print(' ',name,'ERR',repr(e)[:150])

# also compare exp alone
@triton.jit
def ke(X,Y,N,V: tl.constexpr, BLOCK: tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=off<N
    x=tl.load(X+off,mask=m)
    if V==0: y=tl.exp(x)
    elif V==1: y=libdevice.exp(x)
    elif V==2: y=libdevice.exp2(x*1.4426950408889634)
    tl.store(Y+off,y,mask=m)
xe = (torch.rand(N,device=dev)*40-20)
refe = torch.exp(xe)
for v,name in [(0,'tl.exp'),(1,'libdevice.exp'),(2,'libdevice.exp2')]:
    ke[(triton.cdiv(N,1024),)](xe,y,N,v,1024); o=y.clone()
    print(f'  EXP {name:18s} mismatch={(o!=refe).sum().item()/N:.6f}')
# torch sigmoid
refs = torch.sigmoid(xe)
@triton.jit
def ks(X,Y,N,V: tl.constexpr, BLOCK: tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=off<N
    x=tl.load(X+off,mask=m)
    if V==0: y=tl.sigmoid(x)
    elif V==1: y=1.0/(1.0+libdevice.exp(-x))
    elif V==2: y=tl.fdiv(1.0,1.0+libdevice.exp(-x),ieee_rounding=True)
    tl.store(Y+off,y,mask=m)
for v,name in [(0,'tl.sigmoid'),(1,'1/(1+libdev.exp)'),(2,'ieee 1/(1+ld.exp)')]:
    ks[(triton.cdiv(N,1024),)](xe,y,N,v,1024); o=y.clone()
    print(f'  SIGMOID {name:20s} mismatch={(o!=refs).sum().item()/N:.6f}')
