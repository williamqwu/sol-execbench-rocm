import torch, sys, time
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, TOL
def bench(f,n=100):
    for _ in range(20): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
for shape in [(1,16,16),(4,16,16),(1,32,32)]:
    args = make(*shape); tensors=list(args[:32])
    static=[t.clone() for t in tensors]
    nb = sum(t.numel()*4 for t in tensors)
    t=bench(lambda: torch._foreach_copy_(static,tensors))
    print(f"{shape}: copy {t:.4f}ms for {nb/1e6:.1f}MB")
    # pointer stability across repeated make()
    a2 = make(*shape)
    print("   ptr same on re-make:", args[4].data_ptr()==a2[4].data_ptr())
