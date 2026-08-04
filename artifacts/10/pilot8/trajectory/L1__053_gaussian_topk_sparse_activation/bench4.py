import torch, triton, triton.language as tl, time, inspect, sys

@triton.jit
def k_row(X, Y, mult, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    base = row * N
    x = tl.load(X + base + cols).to(tl.float32)
    mean = tl.sum(x, axis=0) * (1.0 / N)
    d = x - mean
    var = tl.sum(d * d, axis=0) * (1.0 / N)
    thr = mean + tl.sqrt(var) * mult
    tl.store(Y + base + cols, tl.maximum(x - thr, 0.0).to(tl.bfloat16))

rows, N = 262, 4096
x = torch.randn(rows, N, device='cuda', dtype=torch.bfloat16)
out = torch.empty_like(x)
mult = -1.28

print(triton.__version__)
ck = k_row.warmup(x, out, mult, N=N, BLOCK=N, num_warps=8, num_stages=1, grid=(1,))
ck._init_handles()
print(inspect.getsource(type(ck).__getattribute__))
