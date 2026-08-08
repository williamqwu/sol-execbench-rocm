import torch, sys, time
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, TOL
import kernel_r as K
from torch.profiler import profile, ProfilerActivity
for shape in [(1,32,32),(4,16,16),(1,16,16)]:
    args=make(*shape); t=list(args[:32]); eps=args[32]
    for _ in range(10): K.run(*args)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(50): K.run(*args)
    cpu=(time.perf_counter()-t0)/50*1e3
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10): K.run(*args)
        torch.cuda.synchronize()
    ka=p.key_averages(); gpu=sum(e.self_device_time_total for e in ka)/10/1000
    nk=sum(e.count for e in ka)/10
    print(f"{shape}: cpu_launch={cpu:.4f} gpu_busy={gpu:.4f} nkernels={nk:.0f}")
