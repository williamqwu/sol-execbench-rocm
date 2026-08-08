import torch, sys, os, time
import torch.nn.functional as F
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import bench
dev='cuda'
C=512
torch.manual_seed(0)
w = (torch.randn(C,C,3,3,device=dev)/24)
b = torch.randn(C,device=dev)

for (B,H,W) in [(1,32,32),(32,32,32),(2,64,64),(1,16,16),(4,48,48)]:
    x = torch.randn(B,C,H,W,device=dev)
    ref = F.conv2d(x,w,b,padding=1)
    t0 = bench(lambda: F.conv2d(x,w,b,padding=1))
    res=[(f"default", t0, 0.0)]
    # channels_last
    xc = x.to(memory_format=torch.channels_last); wc = w.to(memory_format=torch.channels_last)
    o = F.conv2d(xc,wc,b,padding=1)
    res.append(("chan_last", bench(lambda: F.conv2d(xc,wc,b,padding=1)), (o-ref).abs().max().item()))
    # unfold + gemm
    def unf():
        u = F.unfold(x,3,padding=1)
        return (w.view(C,-1)@u + b[None,:,None]).view(B,C,H,W)
    o = unf(); res.append(("unfold_gemm", bench(unf), (o-ref).abs().max().item()))
    # explicit padding + conv
    def padc():
        xp = F.pad(x,(1,1,1,1))
        return F.conv2d(xp,w,b)
    o=padc(); res.append(("pad+conv", bench(padc), (o-ref).abs().max().item()))
    # bias separate
    def nob():
        return F.conv2d(x,w,None,padding=1)+b[None,:,None,None]
    o=nob(); res.append(("conv_nobias", bench(nob), (o-ref).abs().max().item()))
    print(f"--- B{B} {H}x{W}")
    for n,t,d in res:
        print(f"    {n:14s} {t:8.4f} ms  maxdiff={d:.3e} {'EXACT' if d==0 else ''}")
