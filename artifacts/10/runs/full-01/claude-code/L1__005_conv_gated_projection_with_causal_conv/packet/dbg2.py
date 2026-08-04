import torch, torch.nn.functional as F
DEV="cuda:0"; H=2048
torch.manual_seed(0)
B,S=1,64
Bx=torch.randn(B,H,S,dtype=torch.bfloat16,device=DEV)
cw=torch.randn(H,1,4,dtype=torch.bfloat16,device=DEV)
cb=torch.randn(H,dtype=torch.bfloat16,device=DEV)
Bxp=F.pad(Bx,(3,0))
conv=F.conv1d(Bxp,cw,cb,groups=H)

def variant(name, f):
    r=f().to(torch.bfloat16)
    eq=torch.equal(conv,r)
    d=(conv.float()-r.float()).abs()
    print(f"{name:44s} exact={eq}  frac_diff={(conv!=r).float().mean().item():.5f} maxdiff={d.max().item():.4g}")

Bxpf=Bxp.double(); cwd=cw.double(); cbd=cb.double()
def exact():
    a=torch.zeros(B,H,S,device=DEV,dtype=torch.float64)
    for k in range(4): a+=Bxpf[:,:,k:k+S]*cwd[:,0,k][None,:,None]
    return a+cbd[None,:,None]
variant("fp64 exact + bias last", exact)
def exact_bias_first():
    a=cbd[None,:,None].expand(B,H,S).clone()
    for k in range(4): a+=Bxpf[:,:,k:k+S]*cwd[:,0,k][None,:,None]
    return a
variant("fp64 exact, bias first", exact_bias_first)

Bxpf32=Bxp.float(); cwf=cw.float(); cbf=cb.float()
def f32_bias_first():
    a=cbf[None,:,None].expand(B,H,S).clone()
    for k in range(4): a+=Bxpf32[:,:,k:k+S]*cwf[:,0,k][None,:,None]
    return a
variant("fp32 bias-first k=0..3", f32_bias_first)
def f32_bias_last():
    a=torch.zeros(B,H,S,device=DEV,dtype=torch.float32)
    for k in range(4): a+=Bxpf32[:,:,k:k+S]*cwf[:,0,k][None,:,None]
    return a+cbf[None,:,None]
variant("fp32 bias-last k=0..3", f32_bias_last)
def f32_rev():
    a=torch.zeros(B,H,S,device=DEV,dtype=torch.float32)
    for k in [3,2,1,0]: a+=Bxpf32[:,:,k:k+S]*cwf[:,0,k][None,:,None]
    return a+cbf[None,:,None]
variant("fp32 bias-last k=3..0", f32_rev)
def f32_pairs():
    p0=Bxpf32[:,:,0:S]*cwf[:,0,0][None,:,None]+Bxpf32[:,:,1:1+S]*cwf[:,0,1][None,:,None]
    p1=Bxpf32[:,:,2:2+S]*cwf[:,0,2][None,:,None]+Bxpf32[:,:,3:3+S]*cwf[:,0,3][None,:,None]
    return p0+p1+cbf[None,:,None]
variant("fp32 pairwise tree", f32_pairs)
def bf16acc():
    a=torch.zeros(B,H,S,device=DEV,dtype=torch.bfloat16)
    for k in range(4): a=a+(Bxp[:,:,k:k+S]*cw[:,0,k][None,:,None])
    return a+cb[None,:,None]
variant("bf16 accumulate", bf16acc)
