import torch, triton, time
from tune2 import *

def do_bench_style(fn, args, rep=100, warmup=25):
    """Replica of the common do_bench: per-iteration start/end event pairs."""
    for _ in range(warmup): fn(*args)
    torch.cuda.synchronize()
    se=[torch.cuda.Event(True) for _ in range(rep)]
    ee=[torch.cuda.Event(True) for _ in range(rep)]
    for i in range(rep):
        se[i].record(); fn(*args); ee[i].record()
    torch.cuda.synchronize()
    ts=sorted(s.elapsed_time(e)*1e3 for s,e in zip(se,ee))
    return ts[len(ts)//2]

def triton_do_bench(fn,args):
    return triton.testing.do_bench(lambda: fn(*args), warmup=25, rep=100)*1e3

A,B=mk(1)
f2=make_gsk(16,16,64,16,2,2)
tor=lambda a,b: torch.matmul(a,b.T)
for name,f in [("torch",tor),("splitk2launch",f2)]:
    print(f"{name:14s} paired-event={do_bench_style(f,(A,B)):7.2f}  triton.do_bench={triton_do_bench(f,(A,B)):7.2f}  graph={graph_time(f,(A,B)):7.2f}",flush=True)
