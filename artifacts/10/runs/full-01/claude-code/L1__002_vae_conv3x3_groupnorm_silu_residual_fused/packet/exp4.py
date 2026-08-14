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
a=mk(B,H,W)
R=ref(*a)

# --- budget: perturb conv1 output by relative noise r, then continue
x,w1,n1w,n1b,w2,n2w,n2b,eps=a
c1=F.conv2d(x,w1,None,1,1)
print("conv1 std", c1.std().item())
for r in [0,1e-8,3e-8,6e-8,1e-7,3e-7,1e-6]:
    o = c1*(1+ r*torch.randn_like(c1)) if r>0 else c1
    o=F.group_norm(o,32,n1w,n1b,eps); o=F.silu(o)
    o=F.conv2d(o,w2,None,1,1); o=F.group_norm(o,32,n2w,n2b,eps); o=F.silu(o)
    print(f'  perturb conv1 rel={r:.0e} -> matched={matched(o+x,R):.4f}')

# --- budget: perturb conv2 output
o1=F.silu(F.group_norm(c1,32,n1w,n1b,eps))
c2=F.conv2d(o1,w2,None,1,1)
for r in [0,1e-8,3e-8,6e-8,1e-7,3e-7,1e-6]:
    o = c2*(1+ r*torch.randn_like(c2)) if r>0 else c2
    o=F.group_norm(o,32,n2w,n2b,eps); o=F.silu(o)
    print(f'  perturb conv2 rel={r:.0e} -> matched={matched(o+x,R):.4f}')

# --- how accurate is MIOpen conv vs fp64?
c1_64 = F.conv2d(x.double(),w1.double(),None,1,1)
d=(c1.double()-c1_64).abs()
print('MIOpen conv1 err vs fp64: maxabs=%.3e  rms_rel=%.3e (ulp=%.2f)'%(
    d.max().item(), (d/c1_64.abs().clamp(min=1e-3)).pow(2).mean().sqrt().item(),
    (d/ c1.double().abs().clamp(min=1e-6)).median().item()/6e-8))

# --- groupnorm: fp32 torch vs fp64
g_ref = F.group_norm(c1,32,n1w,n1b,eps)
g_64 = F.group_norm(c1.double(),32,n1w.double(),n1b.double(),eps)
d=(g_ref.double()-g_64).abs()
print('torch GN err vs fp64: maxabs=%.3e median_ulp=%.2f'%(d.max().item(), (d/g_ref.double().abs().clamp(min=1e-3)).median().item()/6e-8))

# --- full fp64 conv path but fp32 GN
o=F.conv2d(x.double(),w1.double(),None,1,1).float(); o=F.group_norm(o,32,n1w,n1b,eps); o=F.silu(o)
o=F.conv2d(o.double(),w2.double(),None,1,1).float(); o=F.group_norm(o,32,n2w,n2b,eps); o=F.silu(o)
print('fp64convs+fp32GN matched=%.4f'%matched(o+x,R))

# --- fp32 conv (miopen) + fp64 GN
o=F.conv2d(x,w1,None,1,1); o=F.group_norm(o.double(),32,n1w.double(),n1b.double(),eps).float(); o=F.silu(o)
o=F.conv2d(o,w2,None,1,1); o=F.group_norm(o.double(),32,n2w.double(),n2b.double(),eps).float(); o=F.silu(o)
print('fp32conv+fp64GN matched=%.4f'%matched(o+x,R))
