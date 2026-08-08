import torch
import torch.nn.functional as F
import triton, triton.language as tl
dev='cuda'
x = (torch.rand(1<<22, device=dev)*20-10)   # moderate range
N=x.numel()
ref = torch.exp(x)

def probe(name, fn):
    y = torch.empty_like(x)
    fn[(triton.cdiv(N,1024),)](x,y,N,BLOCK=1024)
    ne=(y!=ref).sum().item()
    rel = ((y-ref).abs()/ref).max().item()
    print(f"  {name:26s} mismatch {ne:8d} ({ne/N*100:5.2f}%) maxrel {rel:.3e}")

@triton.jit
def a0(X,Y,n,BLOCK: tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    tl.store(Y+o, tl.exp(tl.load(X+o,mask=m)), mask=m)
@triton.jit
def a1(X,Y,n,BLOCK: tl.constexpr):
    from triton.language.extra import libdevice
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    tl.store(Y+o, libdevice.exp(tl.load(X+o,mask=m)), mask=m)
@triton.jit
def a2(X,Y,n,BLOCK: tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    tl.store(Y+o, tl.exp2(tl.load(X+o,mask=m)*1.4426950408889634), mask=m)

print("exp (range -10..10):")
for nm,f in [('tl.exp',a0),('libdevice.exp',a1),('exp2',a2)]:
    try: probe(nm,f)
    except Exception as e: print("  ",nm,"ERR",str(e)[:100])

# silu bit-exactness with best exp
print("silu:")
xs = torch.randn(1<<22, device=dev)*3
rs = F.silu(xs)
print("  torch x/(1+exp(-x)):", (xs/(1+torch.exp(-xs)) != rs).sum().item())
@triton.jit
def s0(X,Y,n,BLOCK: tl.constexpr):
    from triton.language.extra import libdevice
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    v=tl.load(X+o,mask=m)
    tl.store(Y+o, v/(1.0+libdevice.exp(-v)), mask=m)
y=torch.empty_like(xs)
s0[(triton.cdiv(N,1024),)](xs,y,N,BLOCK=1024)
print("  triton libdevice silu:", (y!=rs).sum().item(), "maxdiff", (y-rs).abs().max().item())
