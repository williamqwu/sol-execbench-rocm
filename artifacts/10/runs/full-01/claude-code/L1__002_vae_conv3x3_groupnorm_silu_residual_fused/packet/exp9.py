import torch, torch.nn.functional as F, triton, triton.language as tl
dev='cuda:0'; C=256; G=32; D=8
torch.manual_seed(0)
atol,rtol=1.3876e-7,1.1920928955078125e-7

def mk(B,H,W):
    return (torch.randn(B,C,H,W,device=dev), torch.randn(C,C,3,3,device=dev),
            torch.randn(C,device=dev), torch.randn(C,device=dev),
            torch.randn(C,C,3,3,device=dev), torch.randn(C,device=dev), torch.randn(C,device=dev), 1e-6)
def ref(x,w1,n1w,n1b,w2,n2w,n2b,eps):
    o=F.conv2d(x,w1,None,1,1); o=F.group_norm(o,32,n1w,n1b,eps); o=F.silu(o)
    o=F.conv2d(o,w2,None,1,1); o=F.group_norm(o,32,n2w,n2b,eps); o=F.silu(o)
    return o+x
def matched(a,b):
    d=(a.double()-b.double()).abs(); thr=atol+rtol*b.double().abs()
    return (d<=thr).double().mean().item()

@triton.jit
def apply_k(X, MEAN, RSTD, GAMMA, BETA, Y, HxW, RES, C: tl.constexpr, D: tl.constexpr,
            BLOCK: tl.constexpr, SILU: tl.constexpr, ADDRES: tl.constexpr):
    nc = tl.program_id(0); blk = tl.program_id(1)
    c = nc % C; ng = nc // D
    mean = tl.load(MEAN+ng); rstd = tl.load(RSTD+ng)
    gamma = tl.load(GAMMA+c); beta = tl.load(BETA+c)
    a = rstd*gamma; b = -a*mean + beta
    off = blk*BLOCK + tl.arange(0,BLOCK); m = off < HxW
    base = nc.to(tl.int64)*HxW
    x = tl.load(X+base+off, mask=m, other=0.)
    y = a*x + b
    if SILU: y = y / (1.0 + tl.exp(-y))
    if ADDRES: y = y + tl.load(RES+base+off, mask=m, other=0.)
    tl.store(Y+base+off, y, mask=m)

def gn_apply(x, mean, rstd, gamma, beta, silu=True, res=None):
    B=x.shape[0]; HxW=x.shape[2]*x.shape[3]; y=torch.empty_like(x); BLOCK=1024
    apply_k[(B*C, triton.cdiv(HxW,BLOCK))](x,mean,rstd,gamma,beta,y,HxW, res if res is not None else x,
                                            C,D,BLOCK,silu, res is not None, num_warps=4)
    return y

B,H,W=2,64,64
a=mk(B,H,W); x,w1,n1w,n1b,w2,n2w,n2b,eps=a
R=ref(*a)

# ---- pipeline using torch moments + triton apply/silu
def pipe(momfn):
    o=F.conv2d(x,w1,None,1,1)
    m,r = momfn(o); o=gn_apply(o,m,r,n1w,n1b,True)
    o=F.conv2d(o,w2,None,1,1)
    m,r = momfn(o); o=gn_apply(o,m,r,n2w,n2b,True,res=x)
    return o
def torchmom(t):
    _,m,r = torch.native_group_norm(t,n1w,n1b,t.shape[0],C,t.shape[2]*t.shape[3],G,eps); return m,r
print('triton apply + torch moments + triton exp: matched=%.5f'%matched(pipe(torchmom),R))

# ---- isolate: triton apply w/ torch silu
def pipe2():
    o=F.conv2d(x,w1,None,1,1); m,r=torchmom(o); o=F.silu(gn_apply(o,m,r,n1w,n1b,False))
    o=F.conv2d(o,w2,None,1,1); m,r=torchmom(o); o=F.silu(gn_apply(o,m,r,n2w,n2b,False))+x
    return o
print('triton apply + torch moments + TORCH silu: matched=%.5f'%matched(pipe2(),R))

# ---- sensitivity to moment perturbation (torch silu, triton apply)
def pipe3(pm, pr):
    def mm(t):
        m,r=torchmom(t)
        return m*(1+pm*torch.randn_like(m)), r*(1+pr*torch.randn_like(r))
    o=F.conv2d(x,w1,None,1,1); m,r=mm(o); o=F.silu(gn_apply(o,m,r,n1w,n1b,False))
    o=F.conv2d(o,w2,None,1,1); m,r=mm(o); o=F.silu(gn_apply(o,m,r,n2w,n2b,False))+x
    return o
for p in [1e-8,6e-8,3e-7,1e-6,1e-5]:
    print(f'  moments perturbed rel={p:.0e}: matched={matched(pipe3(p,p),R):.5f}')
for p in [1e-8,6e-8,3e-7,1e-6,1e-5]:
    print(f'  RSTD only  perturbed rel={p:.0e}: matched={matched(pipe3(0,p),R):.5f}')
for p in [1e-8,6e-8,3e-7,1e-6,1e-5]:
    print(f'  MEAN only  perturbed rel={p:.0e}: matched={matched(pipe3(p,0),R):.5f}')
