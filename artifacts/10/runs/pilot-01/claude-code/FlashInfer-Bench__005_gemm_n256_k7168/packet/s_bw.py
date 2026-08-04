import torch, triton, triton.language as tl, time
from tune2 import *

@triton.jit
def rd(X, O, n, BLOCK: tl.constexpr):
    pid=tl.program_id(0)
    o=pid*BLOCK+tl.arange(0,BLOCK)
    v=tl.load(X+o, mask=o<n, other=0.0)
    s=tl.sum(v.to(tl.float32))
    tl.store(O+pid, s)

nel=14104*7168
X=torch.randn(nel,device=DEV,dtype=torch.float16)
for BLOCK in [4096,8192,16384]:
    g=triton.cdiv(nel,BLOCK)
    O=torch.empty(g,device=DEV,dtype=torch.float32)
    f=lambda: rd[(g,)](X,O,nel,BLOCK=BLOCK,num_warps=8)
    t=graph_time(f,(),iters=20)
    print(f"pure read BLOCK{BLOCK} grid{g}: {t:.2f}us -> {nel*2/(t*1e-6)/1e9:.0f} GB/s",flush=True)
