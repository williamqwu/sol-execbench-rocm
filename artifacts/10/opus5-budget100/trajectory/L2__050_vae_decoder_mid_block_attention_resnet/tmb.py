import torch, sys, time
import torch.nn.functional as F
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make
def bench(f,n=200):
    for _ in range(50): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
args=make(1,32,32); t=list(args[:32]); C=512
temb=t[1]
wt=torch.cat((t[6],t[26]),0); bt=torch.cat((t[7],t[27]),0)
print("silu:", bench(lambda: F.silu(temb)))
print("linear:", bench(lambda: F.linear(F.silu(temb),wt,bt)))
print("split:", bench(lambda: F.linear(F.silu(temb),wt,bt).split(C,dim=-1)))
tp=F.linear(F.silu(temb),wt,bt)
print("view4d each:", bench(lambda: (tp[:,:C,None,None], tp[:,C:,None,None])))
print("cat:", bench(lambda: (torch.cat((t[6],t[26]),0), torch.cat((t[7],t[27]),0))))
