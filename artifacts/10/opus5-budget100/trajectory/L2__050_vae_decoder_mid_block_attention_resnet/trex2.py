import torch, triton, triton.language as tl
from triton.language.extra import libdevice as ld

@triton.jit
def k(X, Y, n, MODE: tl.constexpr, BLK: tl.constexpr):
    o = tl.program_id(0)*BLK + tl.arange(0,BLK)
    m = o < n
    x = tl.load(X+o, mask=m)
    if MODE == 0: y = tl.exp(x)
    elif MODE == 1: y = ld.exp(x)
    elif MODE == 2: y = tl.exp2(x*1.4426950408889634)
    elif MODE == 3: y = ld.exp2(x*1.4426950408889634)
    else: y = tl.math.exp(x)
    tl.store(Y+o, y, mask=m)

torch.manual_seed(0)
x = (torch.rand(1<<22, device='cuda')*40-20)
ref = torch.exp(x)
for mode in range(5):
    y = torch.empty_like(x)
    k[(triton.cdiv(x.numel(),1024),)](x,y,x.numel(),mode,1024)
    d=(y!=ref); print("exp mode",mode,"mismatch",d.sum().item())
