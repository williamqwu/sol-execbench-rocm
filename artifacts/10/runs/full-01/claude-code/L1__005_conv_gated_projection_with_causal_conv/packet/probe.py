import torch, triton, triton.language as tl
print("has tl.gather:", hasattr(tl, "gather"))
print("has tl.split:", hasattr(tl, "split"))
print("has tl.interleave:", hasattr(tl, "interleave"))
import importlib
print("num_stages/num_warps ok")
dev='cuda:0'
torch.manual_seed(0)
H=2048
import torch.nn.functional as F
import time

def bench(fn, n=50, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Event(True); en=torch.cuda.Event(True)
    st.record()
    for _ in range(n): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/n*1000  # us

for (B,S) in [(1,256),(2,4096),(1,8192),(32,256)]:
    M=B*S
    x=torch.randn(M,H,dtype=torch.bfloat16,device=dev)
    W1=torch.randn(3*H,H,dtype=torch.bfloat16,device=dev)
    b1=torch.randn(3*H,dtype=torch.bfloat16,device=dev)
    W2=torch.randn(H,H,dtype=torch.bfloat16,device=dev)
    b2=torch.randn(H,dtype=torch.bfloat16,device=dev)
    y=torch.randn(M,H,dtype=torch.bfloat16,device=dev)
    t1=bench(lambda: F.linear(x,W1,b1))
    t2=bench(lambda: F.linear(y,W2,b2))
    print(f"M={M:6d}  gemm1={t1:8.1f}us  gemm2={t2:8.1f}us  sum={t1+t2:8.1f}us")
