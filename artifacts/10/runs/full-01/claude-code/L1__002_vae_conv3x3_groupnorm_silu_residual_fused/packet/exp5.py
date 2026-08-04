import torch, torch.nn.functional as F
dev='cuda:0'; C=256
torch.manual_seed(0)
atol,rtol=1.3876e-7,1.1920928955078125e-7

def mk(B,H,W):
    return (torch.randn(B,C,H,W,device=dev), torch.randn(C,C,3,3,device=dev),
            torch.randn(C,device=dev), torch.randn(C,device=dev),
            torch.randn(C,C,3,3,device=dev), torch.randn(C,device=dev), torch.randn(C,device=dev), 1e-6)

def ref(x,w1,n1w,n1b,w2,n2w,n2b,eps):
    r=x
    o=F.conv2d(x,w1,None,1,1); o=F.group_norm(o,32,n1w,n1b,eps); o=F.silu(o)
    o=F.conv2d(o,w2,None,1,1); o=F.group_norm(o,32,n2w,n2b,eps); o=F.silu(o)
    return o+r
def matched(a,b):
    d=(a.double()-b.double()).abs(); thr=atol+rtol*b.double().abs()
    return (d<=thr).double().mean().item()

B,H,W=2,64,64
a=mk(B,H,W); x,w1,n1w,n1b,w2,n2w,n2b,eps=a
R=ref(*a)

c1=F.conv2d(x,w1,None,1,1)
# torch's own mean/rstd
y_t, mean_t, rstd_t = torch.native_group_norm(c1,n1w,n1b,B,C,H*W,32,eps)
print('native_group_norm matches F.group_norm bitwise:', torch.equal(y_t, F.group_norm(c1,32,n1w,n1b,eps)))

# fp64 exact mean/rstd
g = c1.double().reshape(B,32,-1)
m64 = g.mean(-1); v64 = g.var(-1, unbiased=False); r64=(v64+eps).rsqrt()
print('mean torch vs fp64: max rel err', ((mean_t.double()-m64).abs()/m64.abs().clamp(min=1e-9)).max().item())
print('mean abs err', (mean_t.double()-m64).abs().max().item(), ' typical |mean|', m64.abs().mean().item())
print('rstd torch vs fp64: max rel err', ((rstd_t.double()-r64).abs()/r64.abs()).max().item(), '=ulps', ((rstd_t.double()-r64).abs()/r64.abs()).max().item()/6e-8)

# reconstruct torch's GN formula exactly using its mean/rstd
def gn_manual(t, mean, rstd, gamma, beta, B, C):
    scale = rstd.reshape(B,32,1) * gamma.reshape(1,32,C//32)   # a per (b,c)
    bias  = -scale*mean.reshape(B,32,1) + beta.reshape(1,32,C//32)
    return (t.reshape(B,32,C//32,-1) * scale.unsqueeze(-1) + bias.unsqueeze(-1)).reshape(t.shape)
y_m = gn_manual(c1, mean_t, rstd_t, n1w, n1b, B, C)
print('manual GN formula bitwise equal to torch:', torch.equal(y_m, y_t), ' maxdiff', (y_m-y_t).abs().max().item())

# now with fp64 moments cast to fp32
y_m64 = gn_manual(c1, m64.float(), r64.float(), n1w, n1b, B, C)
print('manual GN w/ fp64 moments: maxdiff vs torch', (y_m64-y_t).abs().max().item(), 'ulp~', ((y_m64-y_t).abs()/y_t.abs().clamp(min=1e-2)).median().item()/6e-8)

# end-to-end using fp64 moments in both GNs
def pipe_moment64(x,w1,n1w,n1b,w2,n2w,n2b,eps):
    def gn(t,gm,bt):
        gg=t.double().reshape(B,32,-1); mm=gg.mean(-1); vv=gg.var(-1,unbiased=False); rr=(vv+eps).rsqrt()
        return gn_manual(t, mm.float(), rr.float(), gm, bt, B, C)
    o=F.conv2d(x,w1,None,1,1); o=gn(o,n1w,n1b); o=F.silu(o)
    o=F.conv2d(o,w2,None,1,1); o=gn(o,n2w,n2b); o=F.silu(o)
    return o+x
print('E2E fp64-moment GN matched=%.4f'%matched(pipe_moment64(*a),R))

# end-to-end using torch's moments but manual formula
def pipe_torchmoment(x,w1,n1w,n1b,w2,n2w,n2b,eps):
    def gn(t,gm,bt):
        _,mm,rr = torch.native_group_norm(t,gm,bt,B,C,H*W,32,eps)
        return gn_manual(t,mm,rr,gm,bt,B,C)
    o=F.conv2d(x,w1,None,1,1); o=gn(o,n1w,n1b); o=F.silu(o)
    o=F.conv2d(o,w2,None,1,1); o=gn(o,n2w,n2b); o=F.silu(o)
    return o+x
print('E2E torch-moment manual-formula matched=%.4f'%matched(pipe_torchmoment(*a),R))

# silu form check:  torch silu = x*sigmoid(x)
t=c1
print('silu == x*sigmoid(x) bitwise:', torch.equal(F.silu(t), t*torch.sigmoid(t)))
print('silu == x/(1+exp(-x)) bitwise:', torch.equal(F.silu(t), t/(1+torch.exp(-t))))
