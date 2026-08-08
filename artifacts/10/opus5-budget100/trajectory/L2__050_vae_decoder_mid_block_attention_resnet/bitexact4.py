import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

dev='cuda'
x = torch.randn(1<<22, device=dev)*3
N = x.numel()
ref_exp = torch.exp(x)
ref_silu = F.silu(x)

@triton.jit
def k_tlexp(X,Y,n,BLOCK: tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    tl.store(Y+o, tl.exp(tl.load(X+o,mask=m)), mask=m)

@triton.jit
def k_ldexp(X,Y,n,BLOCK: tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    tl.store(Y+o, libdevice.exp(tl.load(X+o,mask=m)), mask=m)

@triton.jit
def k_silu_ld(X,Y,n,BLOCK: tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    v=tl.load(X+o,mask=m)
    tl.store(Y+o, v/(1.0+libdevice.exp(-v)), mask=m)

@triton.jit
def k_silu_tl(X,Y,n,BLOCK: tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    v=tl.load(X+o,mask=m)
    tl.store(Y+o, v/(1.0+tl.exp(-v)), mask=m)

def run(k, ref, **kw):
    y=torch.empty_like(x)
    k[(triton.cdiv(N,1024),)](x,y,N,BLOCK=1024, **kw)
    ne=(y!=ref).sum().item()
    return ne, (y-ref).abs().max().item()

for nm,k,r in [('tl.exp',k_tlexp,ref_exp),('libdevice.exp',k_ldexp,ref_exp),
               ('silu libdevice',k_silu_ld,ref_silu),('silu tl.exp',k_silu_tl,ref_silu)]:
    for extra in [{}, {'num_warps':4}]:
        try:
            ne,mx = run(k,r,**extra)
            print(f"  {nm:18s} {str(extra):18s} mismatch {ne:8d} ({ne/N*100:5.2f}%) max {mx:.3e}")
        except Exception as e:
            print(f"  {nm:18s} ERR {str(e)[:80]}")
