import torch, sys, time
import torch.nn.functional as F
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, TOL
def bench(f,n=60):
    for _ in range(20): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
for shape in TOL:
    args=make(*shape); t=list(args[:32]); B,H,W=shape; C=512; S=H*W
    h=torch.randn(B,C,S,device='cuda').transpose(1,2)
    w3=torch.cat((t[14],t[16],t[18]),0); b3=torch.cat((t[15],t[17],t[19]),0)
    rq=F.linear(h,t[14],t[15])
    fq=F.linear(h,w3,b3).split(C,dim=-1)[0]
    tf=bench(lambda: F.linear(h,w3,b3))
    ts=bench(lambda: (F.linear(h,t[14],t[15]),F.linear(h,t[16],t[17]),F.linear(h,t[18],t[19])))
    print(f"{shape}: fused={tf:.4f} sep={ts:.4f} best={'fused' if tf<ts else 'SEP'} exact={(fq==rq).all().item()}")
