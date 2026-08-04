import torch, time, math
import torch.nn.functional as F
dev='cuda:0'
def bench(f, n=10):
    for _ in range(3): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3

B=16; T=1500; C=5120; FF=20480; H=20; D=256
M=B*T

print("=== aiter FA numerics (BSHD) ===")
torch.manual_seed(0)
qb=(torch.randn(B,T,H,D,device=dev,dtype=torch.bfloat16))
kb=(torch.randn(B,T,H,D,device=dev,dtype=torch.bfloat16))
vb=(torch.randn(B,T,H,D,device=dev,dtype=torch.bfloat16))
sc=D**-0.5
qt,kt,vt=[x.transpose(1,2).contiguous() for x in (qb,kb,vb)]
aw=torch.matmul(qt*sc,kt.transpose(2,3))
aw=F.softmax(aw,dim=-1,dtype=torch.float32).to(torch.bfloat16)
ref=torch.matmul(aw,vt).transpose(1,2)
del aw; torch.cuda.empty_cache()
from aiter import flash_attn_func
o=flash_attn_func(qb,kb,vb,softmax_scale=sc)
print("  aiter out shape",o.shape,"maxerr vs ref",(o.float()-ref.float()).abs().max().item())
print("  ref absmax",ref.float().abs().max().item())
t=bench(lambda: flash_attn_func(qb,kb,vb,softmax_scale=sc))
print(f"  aiter {t:.3f}ms  {4*B*H*T*T*D/1e12/t*1e3:.1f} TF")
del qb,kb,vb,qt,kt,vt,ref,o; torch.cuda.empty_cache()

print("=== gelu epilogue numerics ===")
a=torch.randn(4096,512,device=dev,dtype=torch.bfloat16)
w=torch.randn(512,512,device=dev,dtype=torch.bfloat16)
bb=torch.randn(512,device=dev,dtype=torch.bfloat16)
r1=F.gelu(F.linear(a,w,bb))
r2=torch._addmm_activation(bb,a,w.t(),use_gelu=True)
print("  addmm_act vs gelu(linear) maxdiff:",(r1.float()-r2.float()).abs().max().item())
r3=F.gelu(F.linear(a,w,bb),approximate='tanh')
print("  tanh-gelu vs exact maxdiff:",(r1.float()-r3.float()).abs().max().item())
del a,w,bb,r1,r2,r3; torch.cuda.empty_cache()

print("=== fc2 layout variants (M=24000,N=5120,K=20480) ===")
fl=2*M*C*FF/1e12
h=torch.randn(M,FF,device=dev,dtype=torch.bfloat16)
w2=torch.randn(C,FF,device=dev,dtype=torch.bfloat16)
t=bench(lambda: torch.mm(h,w2.t())); print(f"  h @ w2.T (NT)   {t:.3f}ms {fl/t*1e3:.0f} TF")
w2n=w2.t().contiguous()
t=bench(lambda: torch.mm(h,w2n)); print(f"  h @ w2n (NN)    {t:.3f}ms {fl/t*1e3:.0f} TF")
del w2n; torch.cuda.empty_cache()
print("=== fc1 output-transposed chain ===")
x=torch.randn(M,C,device=dev,dtype=torch.bfloat16)
w1=torch.randn(FF,C,device=dev,dtype=torch.bfloat16)
fl1=2*M*FF*C/1e12
t=bench(lambda: torch.mm(x,w1.t())); print(f"  fc1 NT          {t:.3f}ms {fl1/t*1e3:.0f} TF")
xt=x.t().contiguous()
t=bench(lambda: torch.mm(w1,xt)); print(f"  fc1 W@Xt -> (FF,M) {t:.3f}ms {fl1/t*1e3:.0f} TF")
t=bench(lambda: x.t().contiguous()); print(f"  transpose x     {t:.3f}ms")
ht=torch.randn(FF,M,device=dev,dtype=torch.bfloat16)
t=bench(lambda: torch.mm(w2,ht)); print(f"  fc2 W2@Ht->(C,M) {t:.3f}ms {fl/t*1e3:.0f} TF")
del x,w1,xt,ht,h,w2; torch.cuda.empty_cache()

print("=== blas library switch ===")
a=torch.randn(M,C,device=dev,dtype=torch.bfloat16)
w=torch.randn(C,C,device=dev,dtype=torch.bfloat16)
flc=2*M*C*C/1e12
for lib in ["default","hipblaslt","blaslt","ck"]:
    try:
        if lib!="default": torch.backends.cuda.preferred_blas_library(lib)
        t=bench(lambda: torch.mm(a,w.t())); print(f"  {lib:10s} {t:.3f}ms {flc/t*1e3:.0f} TF")
    except Exception as e: print(f"  {lib}: {str(e)[:90]}")
torch.backends.cuda.preferred_blas_library("hipblaslt")
