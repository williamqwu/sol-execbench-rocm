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
stream = torch.cuda.current_stream().cuda_stream

def bench_cpu(f, n=1000):
    for _ in range(50): f()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/n*1e6

print("normal cpu=%.2f" % bench_cpu(lambda: k_row[(rows,)](x,out,mult,N=N,BLOCK=N,num_warps=8,num_stages=1)))

# try ck.run(gridX,gridY,gridZ,stream,function,*args)
run = ck.run
fn = ck.function
pm = ck.packed_metadata
xp = x.data_ptr(); op = out.data_ptr()
import ctypes
for trial in [
  lambda: run(rows,1,1,stream,fn,pm,None,None,None,x,out,mult),
  lambda: run(rows,1,1,stream,fn,pm,None,None,None,xp,op,mult),
  lambda: run(rows,1,1,stream,fn,pm,None,x,out,mult),
  lambda: run(rows,1,1,stream,fn,pm,None,None,x,out,mult),
]:
    try:
        trial(); torch.cuda.synchronize()
        print("OK variant -> cpu=%.2f" % bench_cpu(trial))
        break
    except Exception as e:
        print("fail:", type(e).__name__, str(e)[:150])
