import torch, sys
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make
import kernel_q as reference
from torch.profiler import profile, ProfilerActivity

for shape in [(1,32,32),(32,32,32),(16,32,32)]:
    B,H,W=shape
    args = make(B,H,W)
    for _ in range(5): reference.run(*args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as p:
        for _ in range(10): reference.run(*args)
        torch.cuda.synchronize()
    print("="*20, shape)
    ka = p.key_averages()
    tot = sum(e.self_device_time_total for e in ka)
    for e in sorted(ka, key=lambda x:-x.self_device_time_total)[:14]:
        if e.self_device_time_total<=0: continue
        print(f"  {e.key[:60]:60s} {e.self_device_time_total/10/1000:8.4f} ms  n={e.count/10:.0f}  {e.self_device_time_total/tot*100:5.1f}%")
    print(f"  TOTAL {tot/10/1000:.4f} ms")
