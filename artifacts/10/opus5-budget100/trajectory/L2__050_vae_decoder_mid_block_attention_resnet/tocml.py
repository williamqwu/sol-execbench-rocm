import torch, triton, triton.language as tl

@triton.jit
def k(X, Y, n, BLK: tl.constexpr):
    o = tl.program_id(0)*BLK + tl.arange(0,BLK)
    m = o < n
    x = tl.load(X+o, mask=m)
    y = tl.inline_asm_elementwise(
        "v_mov_b32 $0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(Y+o, tl.exp(y), mask=m)

torch.manual_seed(0)
x=(torch.rand(1<<20,device='cuda')*20-10)
ref=torch.exp(x); y=torch.empty_like(x)
k[(triton.cdiv(x.numel(),1024),)](x,y,x.numel(),1024)
print("asm passthrough+exp mismatch", (y!=ref).sum().item())

# What does ATen use? check exp2-based reconstruction
import math
e2 = torch.exp2(x*math.log2(math.e))
print("torch.exp2 recon mismatch", (e2!=ref).sum().item())
