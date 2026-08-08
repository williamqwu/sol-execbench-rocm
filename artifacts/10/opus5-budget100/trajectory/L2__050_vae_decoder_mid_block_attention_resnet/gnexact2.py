import torch, sys
import torch.nn.functional as F
dev='cuda'
torch.manual_seed(0)
ng=32; eps=1e-6

for (B,C,H,W) in [(1,512,32,32),(4,512,16,16)]:
    x = torch.randn(B,C,H,W,device=dev)
    g = torch.randn(C,device=dev); b = torch.randn(C,device=dev)
    ref = F.group_norm(x, ng, g, b, eps)
    out, mean, rstd = torch.native_group_norm(x, g, b, B, C, H*W, ng, eps)
    cpg = C//ng
    mg = mean.repeat_interleave(cpg, dim=1)   # [B,C]
    rg = rstd.repeat_interleave(cpg, dim=1)
    print(f"=== {B}x{C}x{H}x{W}  native==F {(out!=ref).sum().item()}")
    cands = {
      'a=r*g, b=beta-mean*a': (rg*g[None,:], b[None,:]-mg*(rg*g[None,:])),
      'a=r*g, b=beta-(mean*r)*g': (rg*g[None,:], b[None,:]-(mg*rg)*g[None,:]),
      'a=g*r, b=-mean*r*g+beta': (g[None,:]*rg, -mg*rg*g[None,:]+b[None,:]),
      'a=r*g, b=fma(-mean,a,beta)': (rg*g[None,:], torch.addcmul(b[None,:].expand(B,C).contiguous(), -mg, rg*g[None,:])),
    }
    for nm,(a,bb) in cands.items():
        A = a[:,:,None,None]; Bb = bb[:,:,None,None]
        for ap_nm, y in [('a*x+b', A*x+Bb), ('addcmul', torch.addcmul(Bb.expand_as(x).contiguous(), A.expand_as(x), x))]:
            ne=(y!=ref).sum().item()
            print(f"   {nm:28s} {ap_nm:9s} mismatch {ne:9d}/{ref.numel()} max {(y-ref).abs().max().item():.3e}")
