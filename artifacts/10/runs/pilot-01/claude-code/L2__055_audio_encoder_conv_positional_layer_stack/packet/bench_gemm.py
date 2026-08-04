import torch, time
import torch.nn.functional as F
dev='cuda:0'

def bench(f, n=20):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3

B=16; T=1500; C=5120; FF=20480
M=B*T
shapes=[("conv2_Wa",M,C,2*C),("conv2_W0",M,C,C),("qkv",M,3*C,C),("q",M,C,C),
        ("out",M,C,C),("fc1",M,FF,C),("fc2",M,C,FF),("conv1",B*3000,C,240)]
for name,m,n,k in shapes:
    a=torch.randn(m,k,device=dev,dtype=torch.bfloat16)
    w=torch.randn(n,k,device=dev,dtype=torch.bfloat16)
    b=torch.randn(n,device=dev,dtype=torch.bfloat16)
    t=bench(lambda: F.linear(a,w,b))
    fl=2*m*n*k/1e12
    print(f"{name:10s} M={m} N={n} K={k}  {t:7.3f}ms  {fl/t*1e3:7.1f} TFLOPS")
    try:
        t2=bench(lambda: torch._addmm_activation(b,a,w.t(),use_gelu=True))
        print(f"    +gelu epilogue: {t2:7.3f}ms  {fl/t2*1e3:7.1f} TFLOPS")
    except Exception as e:
        print("    addmm_activation:",str(e)[:120])
    del a,w,b; torch.cuda.empty_cache()
