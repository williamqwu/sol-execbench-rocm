import torch, sys, time
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, TOL
import kernel_q as K
def bench(f,n=40):
    for _ in range(12): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
for shape in [(1,16,16),(1,32,32),(4,16,16),(2,41,41)]:
    args=make(*shape); t=list(args[:32]); eps=args[32]
    f=K._fused(t)
    tc=bench(lambda: K._fused(t))
    tfull=bench(lambda: K._body(*t,eps,*K._fused(t)))
    tcached=bench(lambda: K._body(*t,eps,*f))
    print(f"{shape}: cat={tc:.4f} full={tfull:.4f} cached={tcached:.4f} save={tfull-tcached:.4f} ({(tfull-tcached)/tfull*100:.1f}%)")
