import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference, impl
from torch.profiler import profile, ProfilerActivity
torch.manual_seed(0)
for M in [384, 4096]:
    ins = reference.get_inputs({'num_tokens':M}, torch.device('cuda'))
    for _ in range(10): impl.moe(**ins)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(20): impl.moe(**ins)
        torch.cuda.synchronize()
    print(f"=== M={M}")
    ka = p.key_averages()
    for e in sorted(ka, key=lambda x: -x.self_device_time_total)[:8]:
        if e.self_device_time_total>0:
            print(f"  {e.key[:50]:50s} {e.self_device_time_total/20:.1f}us")
