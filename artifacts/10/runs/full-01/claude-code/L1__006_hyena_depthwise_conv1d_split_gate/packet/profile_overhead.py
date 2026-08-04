"""Where does the ~20us floor go? Separate CPU launch overhead from GPU time."""
import torch, triton, time
import triton.language as tl

B, S, D = 1, 512, 256


@triton.jit
def _noop(p):
    pass


def wall(fn, iters=500):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def cpu_only(fn, iters=500):
    """CPU-side dispatch cost: no sync inside the loop."""
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = (time.perf_counter() - t) / iters * 1e6
    torch.cuda.synchronize()
    return dt


x = torch.empty((B, D, S), device="cuda")
u = torch.randn((B, 768, S), device="cuda")

print(f"{'op':45s} {'cpu_us':>9s} {'wall_us':>9s}")
print(f"{'torch.empty (B,256,S) x3':45s} "
      f"{cpu_only(lambda: [torch.empty((B,D,S),device='cuda') for _ in range(3)]):9.2f} "
      f"{'-':>9s}")
print(f"{'torch.empty (3,B,256,S) x1':45s} "
      f"{cpu_only(lambda: torch.empty((3,B,D,S),device='cuda')):9.2f} {'-':>9s}")
print(f"{'u.contiguous() (already contig)':45s} "
      f"{cpu_only(lambda: u.contiguous()):9.2f} {'-':>9s}")

g = (1,)
print(f"{'triton noop launch grid=(1,)':45s} "
      f"{cpu_only(lambda: _noop[g](x)):9.2f} {wall(lambda: _noop[g](x)):9.2f}")
g2 = (4, 256, 1)
print(f"{'triton noop launch grid=(4,256,1)':45s} "
      f"{cpu_only(lambda: _noop[g2](x)):9.2f} {wall(lambda: _noop[g2](x)):9.2f}")

# pre-bound launcher: skip the JIT dispatch path
try:
    ck = _noop.warmup(x, grid=g)
    ck._init_handles()
    print("warmup/_init_handles available")
except Exception as e:
    print("warmup path:", type(e).__name__, e)

import kernel
print(f"{'kernel.run full':45s} "
      f"{cpu_only(lambda: kernel.run(u, torch.randn((768,1,3),device='cuda'), torch.randn(768,device='cuda'))):9.2f}")

w = torch.randn((768, 1, 3), device="cuda")
bi = torch.randn((768,), device="cuda")
print(f"{'kernel.run (preallocated args)':45s} "
      f"{cpu_only(lambda: kernel.run(u, w, bi)):9.2f} {wall(lambda: kernel.run(u, w, bi)):9.2f}")

# large shape: is it GPU bound?
uL = torch.randn((4, 768, 4096), device="cuda")
print(f"{'kernel.run B=4 S=4096':45s} "
      f"{cpu_only(lambda: kernel.run(uL, w, bi)):9.2f} {wall(lambda: kernel.run(uL, w, bi)):9.2f}")
nbytes = 2 * 4 * 768 * 4096 * 4
print(f"   -> {nbytes/(wall(lambda: kernel.run(uL,w,bi))*1e-6)/1e9:.0f} GB/s at wall time")
