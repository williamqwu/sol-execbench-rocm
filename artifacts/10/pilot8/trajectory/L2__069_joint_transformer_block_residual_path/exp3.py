import torch, math, sys
import torch.nn.functional as F
sys.path.insert(0, '/var/tmp/solbench/agent/pilot8/L2__069_joint_transformer_block_residual_path')
import reference as R
from exp2 import run_half, bench

dev = torch.device('cuda')

for b, s_, c in [(1, 1024, 77), (64, 128, 77), (1, 8192, 77)]:
    inp = R.get_inputs({'batch_size': b, 'seq_len': s_, 'context_len': c}, dev)
    print("=== ", b, s_, c, " half:", bench(lambda: run_half(**inp)))
    from torch.profiler import profile, ProfilerActivity
    for _ in range(3):
        run_half(**inp)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            run_half(**inp)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))
