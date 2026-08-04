import torch, time
import torch.nn.functional as F
dev='cuda:0'
def bench(f, n=10):
    for _ in range(3): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3

B=16; T=1500; C=5120; FF=20480; H=20; D=256
M=B*T
print("=== ATTENTION ===")
flops = 4*B*H*T*T*D/1e12
q=torch.randn(B,H,T,D,device=dev,dtype=torch.bfloat16)
k=torch.randn(B,H,T,D,device=dev,dtype=torch.bfloat16)
v=torch.randn(B,H,T,D,device=dev,dtype=torch.bfloat16)
from torch.nn.attention import sdpa_kernel, SDPBackend
for name,bk in [("flash",SDPBackend.FLASH_ATTENTION),("mem_eff",SDPBackend.EFFICIENT_ATTENTION),("cudnn",SDPBackend.CUDNN_ATTENTION)]:
    try:
        with sdpa_kernel(bk):
            t=bench(lambda: F.scaled_dot_product_attention(q,k,v,scale=D**-0.5))
        print(f"  sdpa/{name:8s} {t:7.3f}ms {flops/t*1e3:7.1f} TFLOPS")
    except Exception as e: print(f"  sdpa/{name}: {str(e)[:100]}")
# BSHD layout
qb=torch.randn(B,T,H,D,device=dev,dtype=torch.bfloat16)
try:
    from aiter import flash_attn_func
    t=bench(lambda: flash_attn_func(qb,qb,qb,softmax_scale=D**-0.5))
    print(f"  aiter.fa    {t:7.3f}ms {flops/t*1e3:7.1f} TFLOPS")
except Exception as e: print("  aiter:",str(e)[:200])
try:
    import flash_attn
    t=bench(lambda: flash_attn.flash_attn_func(qb,qb,qb,softmax_scale=D**-0.5))
    print(f"  flash_attn  {t:7.3f}ms {flops/t*1e3:7.1f} TFLOPS")
except Exception as e: print("  flash_attn:",str(e)[:150])
del q,k,v,qb; torch.cuda.empty_cache()

print("=== FC2 variants ===")
h=torch.randn(M,FF,device=dev,dtype=torch.bfloat16)
w2=torch.randn(C,FF,device=dev,dtype=torch.bfloat16)
b2=torch.randn(C,device=dev,dtype=torch.bfloat16)
fl=2*M*C*FF/1e12
t=bench(lambda: F.linear(h,w2,b2)); print(f"  linear      {t:7.3f}ms {fl/t*1e3:7.1f}")
ht=h.t().contiguous()
t=bench(lambda: torch.mm(w2,ht)); print(f"  W@H^T (NT)  {t:7.3f}ms {fl/t*1e3:7.1f}")
del ht; torch.cuda.empty_cache()
# split-N/M chunking
t=bench(lambda: torch.cat([F.linear(h[i*M//2:(i+1)*M//2],w2,b2) for i in range(2)]))
print(f"  2-chunk M   {t:7.3f}ms {fl/t*1e3:7.1f}")
del h,w2,b2; torch.cuda.empty_cache()

print("=== QKV fusion ===")
x=torch.randn(M,C,device=dev,dtype=torch.bfloat16)
wq=torch.randn(3*C,C,device=dev,dtype=torch.bfloat16)
bq=torch.randn(3*C,device=dev,dtype=torch.bfloat16)
t=bench(lambda: F.linear(x,wq,bq)); print(f"  fused qkv   {t:7.3f}ms {2*M*3*C*C/1e12/t*1e3:7.1f}")
wq1=wq[:C].contiguous()
t=bench(lambda: [F.linear(x,wq1,bq[:C]) for _ in range(3)]); print(f"  3 separate  {t:7.3f}ms")
