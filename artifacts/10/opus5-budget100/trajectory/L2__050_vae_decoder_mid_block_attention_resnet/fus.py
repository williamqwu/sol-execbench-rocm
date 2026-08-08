import torch, sys, time
import torch.nn.functional as F
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, TOL
def bench(f,n=40):
    for _ in range(12): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
for shape in TOL:
    args=make(*shape); t=list(args[:32])
    temb=t[1]; C=512
    st=F.silu(temb)
    wt=torch.cat((t[6],t[26]),0); bt=torch.cat((t[7],t[27]),0)
    t_fus=bench(lambda: F.linear(F.silu(temb),wt,bt).split(C,dim=-1))
    t_sep=bench(lambda: (F.linear(F.silu(temb),t[6],t[7]), F.linear(F.silu(temb),t[26],t[27])))
    t_cat=bench(lambda: (torch.cat((t[6],t[26]),0), torch.cat((t[7],t[27]),0)))
    print(f"{shape}: temb fused(no cat)={t_fus:.4f} sep={t_sep:.4f} catcost={t_cat:.4f}")
    break
# qkv
for shape in TOL:
    args=make(*shape); t=list(args[:32]); B,H,W=shape; C=512; S=H*W
    h=torch.randn(B,S,C,device='cuda')
    w3=torch.cat((t[14],t[16],t[18]),0); b3=torch.cat((t[15],t[17],t[19]),0)
    tf=bench(lambda: F.linear(h,w3,b3).split(C,dim=-1))
    ts=bench(lambda: (F.linear(h,t[14],t[15]),F.linear(h,t[16],t[17]),F.linear(h,t[18],t[19])))
    tc=bench(lambda: (torch.cat((t[14],t[16],t[18]),0),torch.cat((t[15],t[17],t[19]),0)))
    print(f"{shape}: qkv fused={tf:.4f} sep={ts:.4f} cat={tc:.4f} net={ts-(tf+tc):+.4f}")
