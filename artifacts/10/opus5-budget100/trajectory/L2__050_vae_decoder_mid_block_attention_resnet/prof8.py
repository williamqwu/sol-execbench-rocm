import torch, sys, time
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, bench
import kernel_q as reference
from torch.profiler import profile, ProfilerActivity

for shape in [(1,16,16),(4,16,16),(1,32,32),(1,48,48),(2,41,41),(1,61,61)]:
    B,H,W=shape
    args = make(B,H,W)
    for _ in range(10): reference.run(*args)
    torch.cuda.synchronize()
    wall = bench(lambda: reference.run(*args), 30, 10)
    # CPU-only launch time (async)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(30): reference.run(*args)
    cpu=(time.perf_counter()-t0)/30*1000
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10): reference.run(*args)
        torch.cuda.synchronize()
    ka=p.key_averages()
    gpu = sum(e.self_device_time_total for e in ka)/10/1000
    nk = sum(e.count for e in ka)/10
    print(f"{shape}: wall={wall:.4f} cpu_launch={cpu:.4f} gpu_busy={gpu:.4f} nkernels={nk:.0f}")
