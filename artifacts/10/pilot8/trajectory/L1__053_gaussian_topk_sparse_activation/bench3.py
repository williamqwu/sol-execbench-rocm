import torch, triton, triton.language as tl, time, sys

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

# warm
for _ in range(20): k_row[(rows,)](x,out,mult,N=N,BLOCK=N,num_warps=8,num_stages=1)
torch.cuda.synchronize()

# CPU wall time per launch (async)
t0=time.perf_counter()
for _ in range(500): k_row[(rows,)](x,out,mult,N=N,BLOCK=N,num_warps=8,num_stages=1)
torch.cuda.synchronize()
t1=time.perf_counter()
print(f"triton normal launch: {(t1-t0)/500*1e6:.2f} us/call (cpu, incl gpu)")

# torch.empty_like cost
t0=time.perf_counter()
for _ in range(2000): o=torch.empty_like(x)
t1=time.perf_counter()
print(f"empty_like: {(t1-t0)/2000*1e6:.2f} us")

# gpu-only event time
def bench(fn,n=200):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    st,en=torch.cuda.Event(True),torch.cuda.Event(True)
    st.record()
    for _ in range(n): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/n*1000
print(f"gpu event per launch: {bench(lambda: k_row[(rows,)](x,out,mult,N=N,BLOCK=N,num_warps=8,num_stages=1)):.2f} us")

# empty kernel launch floor
@triton.jit
def k_nop(X):
    pass
for _ in range(20): k_nop[(1,)](x)
print(f"nop kernel gpu: {bench(lambda: k_nop[(1,)](x)):.2f} us")
t0=time.perf_counter()
for _ in range(1000): k_nop[(1,)](x)
torch.cuda.synchronize()
t1=time.perf_counter()
print(f"nop kernel cpu: {(t1-t0)/1000*1e6:.2f} us")

# low-level launch: reuse compiled kernel
ck = k_row.warmup(x,out,mult,N=N,BLOCK=N,num_warps=8,num_stages=1, grid=(1,))
ck._init_handles()
print("compiled:", type(ck))
import inspect
try:
    print(inspect.signature(ck.run))
except Exception as e: print(e)
try:
    print(inspect.signature(ck.launch_metadata))
except Exception as e: print(e)
