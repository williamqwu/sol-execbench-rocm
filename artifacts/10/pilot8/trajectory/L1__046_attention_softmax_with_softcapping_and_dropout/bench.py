import sys, importlib, time
import torch
import reference

SHAPES = [
    (1, 691, 691), (4, 512, 512), (2, 2048, 2048), (4, 1, 853),
    (64, 128, 128), (4, 256, 256), (16, 256, 256), (8, 256, 256),
    (1, 293, 293), (8, 128, 128), (1, 2048, 2048), (32, 128, 128),
    (1, 512, 512), (2, 1024, 1024), (4, 1024, 1024), (8, 512, 512),
]

mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "kernel_tri")


def bench(fn, x, iters=50, warmup=10):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn(x)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


tot_r = tot_m = 0.0
import math
gm = 0.0
for (b, q, k) in SHAPES:
    x = torch.randn(b, 8, q, k, device='cuda', dtype=torch.bfloat16) * 12.0
    ref = reference.run(x)
    out = mod.run(x)
    err = (out.float() - ref.float()).abs().max().item()
    rel = ((out.float() - ref.float()).abs() / (ref.float().abs() + 1e-6)).max().item()
    tr = bench(reference.run, x)
    tm = bench(mod.run, x)
    gm += math.log(tr / tm)
    print(f"b={b:3d} q={q:5d} k={k:5d}  ref={tr*1000:8.1f}us  mine={tm*1000:8.1f}us  "
          f"sp={tr/tm:5.2f}x  maxabs={err:.3e} maxrel={rel:.3e}")
print(f"geomean speedup: {math.exp(gm/len(SHAPES)):.3f}x")
