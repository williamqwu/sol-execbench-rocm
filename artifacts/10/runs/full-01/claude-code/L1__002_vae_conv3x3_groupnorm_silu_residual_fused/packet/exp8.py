import torch, triton, triton.language as tl
dev='cuda:0'
torch.manual_seed(0)

@triton.jit
def k(X, Y, N, V: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    m = off<N
    x = tl.load(X+off, mask=m)
    if V==0: y = tl.exp(-x)
    elif V==1: y = tl.math.exp(-x)
    elif V==2: y = tl.exp2(-x*1.4426950408889634)
    elif V==3: y = tl.sigmoid(x)
    elif V==4: y = 1.0/(1.0+tl.exp(-x))
    elif V==5: y = x/(1.0+tl.exp(-x))
    elif V==6: y = tl.math.exp2(-x*1.4426950408889634)
    elif V==7: y = x*tl.sigmoid(x)
    tl.store(Y+off, y, mask=m)

N=1<<22
x = (torch.rand(N,device=dev)*40-20)
y=torch.empty_like(x)
def run(v):
    k[(triton.cdiv(N,1024),)](x,y,N,v,1024); return y.clone()

import torch.nn.functional as F
te = torch.exp(-x); ts = torch.sigmoid(x); tsl = F.silu(x)
for v,name,ref in [(0,'tl.exp',te),(1,'tl.math.exp',te),(2,'exp2*log2e',te),(6,'tl.math.exp2',te),
                   (3,'tl.sigmoid',ts),(4,'1/(1+exp(-x))',ts),(5,'x/(1+exp(-x))',tsl),(7,'x*sigmoid',tsl)]:
    o=run(v)
    ne=(o!=ref).sum().item()
    d=(o.double()-ref.double()).abs()
    ulp = (d/ref.double().abs().clamp(min=1e-30)/6e-8)
    print(f'{name:16s} mismatch={ne/N:.5f} maxulp={ulp.max().item():.2f} maxabs={d.max().item():.3e}')
