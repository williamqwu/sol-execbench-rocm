from lt import *
import torch, kernel, reference
B,S=1,131
inp=gen(B,S,seed=1131)
# count kernels via profiler
from torch.profiler import profile, ProfilerActivity
for _ in range(5): kernel.run(**inp)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(10): kernel.run(**inp)
    torch.cuda.synchronize()
evs=[e for e in p.key_averages() if e.device_time_total>0]
tot=0; cnt=0
for e in sorted(evs,key=lambda x:-x.device_time_total):
    print(f"  {e.count/10:5.1f}x {e.device_time_total/10:8.2f}us  {e.key[:70]}")
    tot+=e.device_time_total/10; cnt+=e.count/10
print(f"TOTAL {tot:.1f}us across {cnt:.0f} kernels")
print("wall", bench(lambda: kernel.run(**inp), iters=200, warm=50)*1000, "us")
