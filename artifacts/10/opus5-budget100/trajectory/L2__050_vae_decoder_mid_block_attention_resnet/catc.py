import torch, sys, time
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, TOL
import kernel_k
def bench(f,n=50):
    for _ in range(15): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
for shape in [(1,16,16),(1,32,32),(32,32,32)]:
    args=make(*shape); t=list(args[:32]); eps=args[32]
    tc = bench(lambda: kernel_k._fused(t))
    f = kernel_k._fused(t)
    tb = bench(lambda: kernel_k._body(*t, eps, *f))
    tfull = bench(lambda: kernel_k._body(*t, eps, *kernel_k._fused(t)))
    print(f"{shape}: cat={tc:.4f} body={tb:.4f} full={tfull:.4f}  cat share={tc/tfull*100:.1f}%")
