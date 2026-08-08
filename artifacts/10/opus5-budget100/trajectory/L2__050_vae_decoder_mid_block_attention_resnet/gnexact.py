import torch, sys
import torch.nn.functional as F
dev='cuda'
torch.manual_seed(0)

for (B,C,H,W) in [(1,512,32,32),(32,512,32,32),(2,512,64,64),(1,512,16,16)]:
    x = torch.randn(B,C,H,W,device=dev)
    g = torch.randn(C,device=dev); b = torch.randn(C,device=dev)
    eps=1e-6; ng=32
    ref = F.group_norm(x, ng, g, b, eps)
    out, mean, rstd = torch.native_group_norm(x, g, b, B, C, H*W, ng, eps)
    print(f"--- {B}x{C}x{H}x{W}  native==F: {(out!=ref).sum().item()}")
    xg = x.view(B,ng,-1)
    m2 = xg.mean(-1)
    v2 = xg.var(-1, unbiased=False)
    r2 = torch.rsqrt(v2+eps)
    print(f"   mean match: {(m2!=mean).sum().item()}/{mean.numel()}  rstd match: {(r2!=rstd).sum().item()}/{rstd.numel()}")
    # E[x^2]-E[x]^2
    v3 = (xg*xg).mean(-1) - m2*m2
    r3 = torch.rsqrt(v3+eps)
    print(f"   rstd via E[x2]: {(r3!=rstd).sum().item()}")
    # apply with exact mean/rstd from aten
    cpg = C//ng
    a = rstd.repeat_interleave(cpg, dim=1) * g[None,:]
    bb = b[None,:] - mean.repeat_interleave(cpg,dim=1)*a
    y = a[:,:,None,None]*x + bb[:,:,None,None]
    print(f"   apply a*x+b mismatch: {(y!=ref).sum().item()}/{ref.numel()}  max {(y-ref).abs().max().item():.3e}")
    y2 = torch.addcmul(bb[:,:,None,None], a[:,:,None,None], x)
    print(f"   apply fma mismatch: {(y2!=ref).sum().item()}  max {(y2-ref).abs().max().item():.3e}")
    y3 = (x - mean.repeat_interleave(cpg,1)[:,:,None,None]) * rstd.repeat_interleave(cpg,1)[:,:,None,None] * g[None,:,None,None] + b[None,:,None,None]
    print(f"   apply naive mismatch: {(y3!=ref).sum().item()}  max {(y3-ref).abs().max().item():.3e}")
