import torch
import torch.nn.functional as F
import triton, triton.language as tl
dev='cuda'

x = (torch.randn(1<<22, device=dev)*3)
N = x.numel()

def probe(name, jitfn, ref, **kw):
    y = torch.empty_like(x)
    jitfn[(triton.cdiv(N,1024),)](x,y,N,BLOCK=1024,**kw)
    ne=(y!=ref).sum().item()
    print(f"  {name:34s} mismatch {ne:8d}/{N} ({ne/N*100:5.2f}%) max {(y-ref).abs().max().item():.3e}")

# --- exp ---
print("exp vs torch.exp:")
@triton.jit
def e0(X,Y,n,BLOCK: tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    tl.store(Y+o, tl.exp(tl.load(X+o,mask=m)), mask=m)
@triton.jit
def e1(X,Y,n,BLOCK: tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    tl.store(Y+o, tl.math.exp(tl.load(X+o,mask=m)), mask=m)
@triton.jit
def e2(X,Y,n,BLOCK: tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    tl.store(Y+o, tl.exp2(tl.load(X+o,mask=m)*1.4426950408889634), mask=m)
re_ = torch.exp(x)
for nm,f in [('tl.exp',e0),('tl.math.exp',e1),('exp2*log2e',e2)]:
    try: probe(nm,f,re_)
    except Exception as ex: print("  ",nm,"ERR",str(ex)[:90])

try:
    from triton.language.extra import libdevice as ld
    print("  libdevice available:", [a for a in dir(ld) if 'exp' in a or 'sqrt' in a][:12])
except Exception as ex:
    print("  no libdevice:", str(ex)[:80])
try:
    from triton.language.extra.hip import libdevice as hld
    print("  hip libdevice:", [a for a in dir(hld) if 'exp' in a or 'sqrt' in a][:12])
except Exception as ex:
    print("  no hip libdevice:", str(ex)[:80])

# --- rsqrt ---
print("rsqrt vs torch.rsqrt:")
xp = x.abs()+0.5
@triton.jit
def r0(X,Y,n,BLOCK: tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    tl.store(Y+o, tl.rsqrt(tl.load(X+o,mask=m)), mask=m)
@triton.jit
def r1(X,Y,n,BLOCK: tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    tl.store(Y+o, 1.0/tl.sqrt(tl.load(X+o,mask=m)), mask=m)
@triton.jit
def r2(X,Y,n,BLOCK: tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=o<n
    v=tl.load(X+o,mask=m)
    tl.store(Y+o, tl.math.rsqrt(v), mask=m)
rr = torch.rsqrt(xp)
xsave=x; x=xp
for nm,f in [('tl.rsqrt',r0),('1/tl.sqrt',r1),('tl.math.rsqrt',r2)]:
    try: probe(nm,f,rr)
    except Exception as ex: print("  ",nm,"ERR",str(ex)[:90])
x=xsave

# torch variants of rsqrt
print("  torch 1/sqrt vs rsqrt mismatch:", ((1.0/torch.sqrt(xp))!=rr).sum().item())
print("  torch pow(-0.5) mismatch:", ((xp**-0.5)!=rr).sum().item())
