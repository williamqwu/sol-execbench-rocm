import torch, triton, triton.language as tl
import torch.nn.functional as F

@triton.jit
def mom(X, MU, RS, N, eps, BLK: tl.constexpr):
    i = tl.program_id(0)
    base = i.to(tl.int64)*N
    s = tl.zeros([BLK], tl.float32); s2 = tl.zeros([BLK], tl.float32)
    for off in tl.range(0, N, BLK):
        o = off + tl.arange(0,BLK); m = o < N
        x = tl.load(X+base+o, mask=m, other=0.0)
        s += x; s2 += x*x
    su = tl.sum(s); su2 = tl.sum(s2)
    mu = su/N; var = su2/N - mu*mu
    tl.store(MU+i, mu); tl.store(RS+i, tl.rsqrt(var+eps))

@triton.jit
def app(X, Y, MU, RS, G, B, C, HxW, D, total, BLK: tl.constexpr):
    o = tl.program_id(0).to(tl.int64)*BLK + tl.arange(0,BLK)
    m = o < total
    nc = o // HxW
    c = nc % C
    ng = (nc // C)*(C//D) + c//D
    x = tl.load(X+o, mask=m, other=0.)
    mu = tl.load(MU+ng, mask=m, other=0.); rs = tl.load(RS+ng, mask=m, other=0.)
    g = tl.load(G+c, mask=m, other=0.); b = tl.load(B+c, mask=m, other=0.)
    a = rs*g
    tl.store(Y+o, a*x + (-mu*a+b), mask=m)

def tgn(x, g, b, ng, eps):
    Bs,C = x.shape[0], x.shape[1]; HxW = x.numel()//(Bs*C); D = C//ng
    mu = torch.empty(Bs*ng, device=x.device); rs = torch.empty(Bs*ng, device=x.device)
    N = D*HxW
    mom[(Bs*ng,)](x, mu, rs, N, eps, 1024, num_warps=8)
    y = torch.empty_like(x)
    app[(triton.cdiv(x.numel(),1024),)](x,y,mu,rs,g,b,C,HxW,D,x.numel(),1024)
    return y

torch.manual_seed(0)
for B,H,W in [(1,32,32),(16,32,32),(2,41,41)]:
    x=torch.randn(B,512,H,W,device='cuda'); g=torch.ones(512,device='cuda'); bb=torch.zeros(512,device='cuda')
    r=F.group_norm(x,32,g,bb,1e-6); y=tgn(x,g,bb,32,1e-6)
    print(f"B{B} {H}x{W}: mismatch={(y!=r).sum().item()}/{y.numel()} max={(y-r).abs().max().item():.3e}")
