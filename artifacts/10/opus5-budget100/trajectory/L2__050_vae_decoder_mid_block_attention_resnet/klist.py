import torch, sys, collections
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make
import reference
from torch.profiler import profile, ProfilerActivity

for shape in [(1,16,16),(32,32,32)]:
    B,H,W=shape
    args=make(B,H,W)
    for _ in range(5): reference.run(*args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5): reference.run(*args)
        torch.cuda.synchronize()
    ka=p.key_averages()
    print("="*30, shape)
    tot=0; cnt=0
    for e in sorted(ka,key=lambda x:-x.self_device_time_total):
        if e.count==0: continue
        n=e.count/5
        t=e.self_device_time_total/5/1000
        if t<=0 and n<=0: continue
        tot+=t; cnt+=n
        print(f"  n={n:5.1f} {t*1000:8.1f}us  {e.key[:78]}")
    print(f"  TOTAL {tot:.4f} ms, kernels={cnt:.0f}")
