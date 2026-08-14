import torch, triton
import triton.language as tl

dev="cuda:0"; H=2048
torch.manual_seed(0)

def timeit(fn, iters=200, warmup=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Event(True); en=torch.cuda.Event(True)
    st.record()
    for _ in range(iters): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/iters*1000

@triton.jit
def noop(X):
    pass

x=torch.zeros(1,device=dev)
print(f"empty launch grid1   : {timeit(lambda: noop[(1,)](x)):.2f} us")
print(f"empty launch grid1024: {timeit(lambda: noop[(1024,)](x)):.2f} us")

# peak bf16 gemm
for m,n,k in [(8192,8192,8192),(4096,4096,4096),(8192,2048,2048),(2048,2048,8192)]:
    a=torch.randn(m,k,device=dev,dtype=torch.bfloat16)
    b=torch.randn(k,n,device=dev,dtype=torch.bfloat16)
    t=timeit(lambda: a@b, iters=50)
    fl=2*m*n*k
    print(f"gemm {m}x{n}x{k}: {t:8.1f}us  {fl/t*1e-9:7.2f} TFLOPS")

# bandwidth
for mb in [8,64,256]:
    n=mb*1024*1024//2
    a=torch.randn(n,device=dev,dtype=torch.bfloat16)
    t=timeit(lambda: a.clone(), iters=50)
    print(f"copy {mb}MB: {t:8.1f}us  {2*mb/1024/(t*1e-6)/1024:7.2f} TB/s")

# empty alloc cost
for shp in [(2048,2048),(8192,2048)]:
    t=timeit(lambda: torch.empty(shp,device=dev,dtype=torch.bfloat16))
    print(f"empty{shp}: {t:.2f} us")
