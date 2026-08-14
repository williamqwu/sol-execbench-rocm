import torch, torch.nn.functional as F, time
dev='cuda:0'; C=256
torch.manual_seed(0)

def mk(B,H,W):
    return (torch.randn(B,C,H,W,device=dev), torch.randn(C,C,3,3,device=dev),
            torch.randn(C,device=dev), torch.randn(C,device=dev),
            torch.randn(C,C,3,3,device=dev), torch.randn(C,device=dev), torch.randn(C,device=dev), 1e-6)

def pipeline(conv, x,w1,n1w,n1b,w2,n2w,n2b,eps):
    r=x
    o=conv(x,w1); o=F.group_norm(o,32,n1w,n1b,eps); o=F.silu(o)
    o=conv(o,w2); o=F.group_norm(o,32,n2w,n2b,eps); o=F.silu(o)
    return o+r

conv_ref = lambda a,w: F.conv2d(a,w,None,1,1)

def conv_im2col(a,w):
    B,Ci,H,W_ = a.shape
    col = F.unfold(a,3,padding=1)               # B, Ci*9, H*W
    out = torch.matmul(w.reshape(C,-1), col)     # B, C, H*W
    return out.reshape(B,C,H,W_)

def conv_cl(a,w):
    return F.conv2d(a.contiguous(memory_format=torch.channels_last), w.contiguous(memory_format=torch.channels_last),None,1,1)

def stats(a,b,atol,rtol):
    d=(a.double()-b.double()).abs(); thr=atol+rtol*b.double().abs()
    return (d<=thr).double().mean().item(), d.max().item()

atol,rtol=1.3876e-7,1.1920928955078125e-7
for (B,H,W) in [(2,64,64),(4,128,128)]:
    a=mk(B,H,W)
    ref=pipeline(conv_ref,*a)
    for name,cv in [('channels_last',conv_cl),('im2col_matmul',conv_im2col)]:
        try:
            o=pipeline(cv,*a)
            m,mx=stats(o,ref,atol,rtol)
            print(f'B{B} H{H} W{W} {name}: matched={m:.4f} maxabs={mx:.3e}')
        except Exception as e:
            print(name,'ERR',e)
    # also: conv only comparison
    c_ref = conv_ref(a[0],a[1]); c_i = conv_im2col(a[0],a[1])
    d=(c_ref.double()-c_i.double()).abs()
    print(f'   conv-only im2col diff maxabs={d.max().item():.3e} rel_med={(d/c_ref.double().abs().clamp(min=1e-3)).median().item():.3e} std={c_ref.std().item():.2f}')
