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

ck = k_row.warmup(x, out, mult, N=N, BLOCK=N, num_warps=8, num_stages=1, grid=(1,))
ck._init_handles()
print("attrs:", [a for a in dir(ck) if not a.startswith('__')])
print("run sig ok")
stream = torch.cuda.current_stream().cuda_stream
fn = ck.function
launcher = ck.run

def bench_cpu(f, n=500):
    for _ in range(50): f()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/n*1e6

def bench_gpu(f,n=300):
    for _ in range(50): f()
    torch.cuda.synchronize()
    st,en=torch.cuda.Event(True),torch.cuda.Event(True); st.record()
    for _ in range(n): f()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/n*1000

print("normal   cpu=%.2f gpu=%.2f" % (
  bench_cpu(lambda: k_row[(rows,)](x,out,mult,N=N,BLOCK=N,num_warps=8,num_stages=1)),
  bench_gpu(lambda: k_row[(rows,)](x,out,mult,N=N,BLOCK=N,num_warps=8,num_stages=1))))

# low-level: (gridX,gridY,gridZ,stream,function,*args)
try:
    args = (rows,1,1,stream,fn,ck.packed_metadata if hasattr(ck,'packed_metadata') else None)
    print("packed_metadata:", getattr(ck,'packed_metadata',None))
except Exception as e: print(e)

import triton.runtime.driver as drv
print(ck.metadata)
