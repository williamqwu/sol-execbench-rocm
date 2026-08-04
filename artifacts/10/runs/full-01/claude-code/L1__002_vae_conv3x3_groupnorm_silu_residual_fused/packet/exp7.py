import torch, triton, triton.language as tl, torch.nn.functional as F
dev='cuda:0'; C=256; G=32; D=C//G
torch.manual_seed(0)

@triton.jit
def apply_k(X, MEAN, RSTD, GAMMA, BETA, Y, HxW, C: tl.constexpr, D: tl.constexpr,
            BLOCK: tl.constexpr, SILU: tl.constexpr, VARIANT: tl.constexpr):
    nc = tl.program_id(0)
    blk = tl.program_id(1)
    c = nc % C
    ng = nc // D
    mean = tl.load(MEAN+ng); rstd = tl.load(RSTD+ng)
    gamma = tl.load(GAMMA+c); beta = tl.load(BETA+c)
    a = rstd*gamma
    b = -a*mean + beta
    off = blk*BLOCK + tl.arange(0,BLOCK)
    m = off < HxW
    base = nc.to(tl.int64)*HxW
    x = tl.load(X+base+off, mask=m, other=0.)
    if VARIANT == 0:
        y = a*x + b
    elif VARIANT == 1:
        y = tl.math.fma(a, x, b)
    else:
        y = (x - mean)*rstd*gamma + beta
    if SILU == 1:
        y = y / (1.0 + tl.exp(-y))
    elif SILU == 2:
        y = y * tl.sigmoid(y)
    elif SILU == 3:
        y = y / (1.0 + tl.math.exp(-y))
    tl.store(Y+base+off, y, mask=m)

def gn_apply(x, mean, rstd, gamma, beta, silu, variant):
    B = x.shape[0]; HxW = x.shape[2]*x.shape[3]
    y = torch.empty_like(x)
    BLOCK=1024
    apply_k[(B*C, triton.cdiv(HxW,BLOCK))](x, mean, rstd, gamma, beta, y, HxW, C, D, BLOCK, silu, variant, num_warps=4)
    return y

B,H,W=2,64,64
x = torch.randn(B,C,H,W,device=dev)*47.0
gamma = torch.randn(C,device=dev); beta=torch.randn(C,device=dev)
eps=1e-6
y_t, mean_t, rstd_t = torch.native_group_norm(x,gamma,beta,B,C,H*W,G,eps)

for v in [0,1,2]:
    y = gn_apply(x, mean_t, rstd_t, gamma, beta, 0, v)
    print(f'GN variant {v}: bitexact={torch.equal(y,y_t)} maxdiff={(y-y_t).abs().max().item():.3e}')

# SiLU variants against torch
ys_t = F.silu(y_t)
for s in [1,2,3]:
    y = gn_apply(x, mean_t, rstd_t, gamma, beta, s, 1)
    print(f'GN+SiLU silu-variant {s}: bitexact={torch.equal(y,ys_t)} maxdiff={(y-ys_t).abs().max().item():.3e}')
