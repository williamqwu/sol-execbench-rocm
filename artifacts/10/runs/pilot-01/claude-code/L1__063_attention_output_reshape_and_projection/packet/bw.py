import time, torch, triton
import triton.language as tl

dev = "cuda:0"
K, N = 16384, 7168
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
nbytes = w.numel() * 2
print(f"weight = {nbytes/1e6:.1f} MB")


def bench(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


@triton.jit
def _read(P, n, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros((BLOCK,), tl.float32)
    i = pid * BLOCK
    while i < n:
        acc += tl.load(P + i + tl.arange(0, BLOCK)).to(tl.float32)
        i += nprog * BLOCK
    tl.store(out + pid * BLOCK + tl.arange(0, BLOCK), acc)


n = w.numel()
for nprog in (1024, 2048, 4096, 8192):
    BLOCK = 2048
    o = torch.empty(nprog * BLOCK, device=dev, dtype=torch.float32)
    f = lambda: _read[(nprog,)](w, n, o, BLOCK=BLOCK, num_warps=8)
    t = bench(f)
    print(f"stream-read nprog={nprog:<6} {t:.4f} ms  {nbytes/t*1e-9:7.0f} GB/s")

t = bench(lambda: w.clone())
print(f"clone (r+w)  {t:.4f} ms  {2*nbytes/t*1e-9:7.0f} GB/s")
t = bench(lambda: torch.sum(w.view(torch.int16), dtype=torch.int64))
print(f"sum          {t:.4f} ms  {nbytes/t*1e-9:7.0f} GB/s")
