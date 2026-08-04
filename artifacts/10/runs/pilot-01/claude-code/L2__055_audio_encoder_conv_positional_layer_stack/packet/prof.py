import torch, time, sys
import torch.nn.functional as F
import reference
dev = torch.device('cuda:0')
B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
AX = dict(input_seq_len=3000, output_seq_len=1500, num_mel_bins=80,
          d_model=5120, encoder_ffn_dim=20480, num_heads=20, head_dim=256, batch_size=B)
torch.manual_seed(0)
inp = reference.get_inputs(AX, dev)
import kernel
args = list(inp.values())
for _ in range(3): out = kernel.run(*args)
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(3): kernel.run(*args)
    torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25))
