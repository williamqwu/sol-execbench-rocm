import torch, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference, impl
torch.manual_seed(0)
def bench(f, n=20):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
Ms = [int(x) for x in sys.argv[1:]] or [384,1024,2048,4096]
for M in Ms:
    ins = reference.get_inputs({'num_tokens':M}, torch.device('cuda'))
    ref = reference.run(**ins)
    got = impl.moe(**ins)
    d = (ref.float()-got.float()).abs()
    tr = bench(lambda: reference.run(**ins))
    tg = bench(lambda: impl.moe(**ins))
    print(f"M={M} maxabs={d.max().item():.4f} refmax={ref.abs().max().item():.3f} ref={tr:.4f}ms mine={tg:.4f}ms speedup={tr/tg:.2f}x")
