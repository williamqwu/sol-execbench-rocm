import torch, time
dev='cuda'
def bench(fn,it=30,wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/it
for n in [4096, 8192, 16384]:
    a=torch.randn(n,n,device=dev,dtype=torch.bfloat16)
    b=torch.randn(n,n,device=dev,dtype=torch.bfloat16)
    bt=b.t().contiguous()
    t=bench(lambda: a@bt.t())
    print(n,'NT', f'{2*n**3/t/1e12:.0f} TFLOPS', f'{t*1e6:.0f}us')
    t=bench(lambda: a@b)
    print(n,'NN', f'{2*n**3/t/1e12:.0f} TFLOPS')
# bandwidth
x=torch.empty(1<<28, device=dev, dtype=torch.bfloat16)
y=torch.empty_like(x)
t=bench(lambda: y.copy_(x))
print('copy BW', f'{2*x.numel()*2/t/1e12:.2f} TB/s')
