import torch, time, triton, triton.language as tl
@triton.jit
def nop(X): pass
x=torch.zeros(1,device='cuda')
def bench(f,n=500):
    for _ in range(50): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6
print('nop grid1', bench(lambda: nop[(1,)](x)))
print('nop grid448', bench(lambda: nop[(448,)](x)))
print('nop x5', bench(lambda: [nop[(1,)](x) for _ in range(5)]))
w=torch.randn(3584,2048,device='cuda',dtype=torch.bfloat16)
print('cast', bench(lambda: w.to(torch.float8_e4m3fn)))
print('cast x2', bench(lambda: (w.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn))))
