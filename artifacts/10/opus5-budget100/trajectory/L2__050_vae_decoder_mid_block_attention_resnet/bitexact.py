import torch, sys
import torch.nn.functional as F
import triton, triton.language as tl
dev='cuda'

@triton.jit
def silu_k(X, Y, n, MODE: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    m = o < n
    x = tl.load(X+o, mask=m)
    if MODE == 0:
        y = x / (1.0 + tl.exp(-x))
    elif MODE == 1:
        y = x * tl.sigmoid(x)
    elif MODE == 2:
        y = x / (1.0 + tl.math.exp(-x))
    else:
        y = x / (1.0 + tl.exp2(-x * 1.4426950408889634))
    tl.store(Y+o, y, mask=m)

x = torch.randn(1<<22, device=dev)*3
ref = F.silu(x)
for mode in range(4):
    y = torch.empty_like(x)
    silu_k[(triton.cdiv(x.numel(),1024),)](x,y,x.numel(),MODE=mode,BLOCK=1024)
    ne = (y!=ref).sum().item()
    print(f"silu mode{mode}: mismatch {ne}/{x.numel()} maxdiff {(y-ref).abs().max().item():.3e}")

print("torch x*sigmoid mismatch:", ((x*torch.sigmoid(x))!=ref).sum().item())
print("torch x/(1+exp(-x)) mismatch:", ((x/(1+torch.exp(-x)))!=ref).sum().item())

# exp bit-exactness
@triton.jit
def exp_k(X,Y,n,BLOCK: tl.constexpr):
    o = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    tl.store(Y+o, tl.exp(tl.load(X+o,mask=m)), mask=m)
y = torch.empty_like(x)
exp_k[(triton.cdiv(x.numel(),1024),)](x,y,x.numel(),BLOCK=1024)
print("exp mismatch:", (y!=torch.exp(x)).sum().item())

# sigmoid
@triton.jit
def sig_k(X,Y,n,BLOCK: tl.constexpr):
    o = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    tl.store(Y+o, tl.sigmoid(tl.load(X+o,mask=m)), mask=m)
sig_k[(triton.cdiv(x.numel(),1024),)](x,y,x.numel(),BLOCK=1024)
print("sigmoid mismatch:", (y!=torch.sigmoid(x)).sum().item())
