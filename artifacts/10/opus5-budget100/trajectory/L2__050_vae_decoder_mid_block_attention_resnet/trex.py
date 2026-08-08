import torch, triton, triton.language as tl
import torch.nn.functional as F
from triton.language.extra import libdevice as ld

@triton.jit
def silu_k(X, Y, n, MODE: tl.constexpr, BLK: tl.constexpr):
    o = tl.program_id(0)*BLK + tl.arange(0,BLK)
    m = o < n
    x = tl.load(X+o, mask=m)
    if MODE == 0:
        y = x / (1.0 + tl.exp(-x))
    elif MODE == 1:
        y = x / (1.0 + ld.exp(-x))
    elif MODE == 2:
        y = x * tl.sigmoid(x)
    else:
        y = x / (1.0 + tl.exp2(-x * 1.4426950408889634))
    tl.store(Y+o, y, mask=m)

torch.manual_seed(0)
x = torch.randn(1<<22, device='cuda')*3
ref = F.silu(x)
for mode in range(4):
    y = torch.empty_like(x)
    silu_k[(triton.cdiv(x.numel(),1024),)](x,y,x.numel(),mode,1024)
    print("mode",mode,"mismatch",(y!=ref).sum().item(), (y-ref).abs().max().item())
