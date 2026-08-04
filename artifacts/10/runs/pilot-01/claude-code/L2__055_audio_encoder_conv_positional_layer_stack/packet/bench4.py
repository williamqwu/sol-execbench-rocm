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
print("=== conv2 as single K=15360 GEMM vs split ===")
fl=2*M*C*3*C/1e12
a=torch.randn(M,3*C,device=dev,dtype=torch.bfloat16)
w=torch.randn(C,3*C,device=dev,dtype=torch.bfloat16)
bi=torch.randn(C,device=dev,dtype=torch.bfloat16)
t=bench(lambda: F.linear(a,w,bi)); print(f"  single K=15360 {t:.3f}ms {fl/t*1e3:.0f} TF")
del a,w; torch.cuda.empty_cache()
# im2col build cost: gather 3C from a (B,T1,C) source
g1=torch.randn(B,3000,C,device=dev,dtype=torch.bfloat16)
def build():
    p=torch.empty(M,3*C,device=dev,dtype=torch.bfloat16)
    gv=g1.view(B,1500,2,C)
    p[:,C:3*C].view(B,1500,2*C).copy_(g1.view(B,1500,2*C))
    return p
t=bench(build); print(f"  im2col build   {t:.3f}ms")
del g1; torch.cuda.empty_cache()

print("=== split conv2 (current) ===")
g=torch.randn(M,2*C,device=dev,dtype=torch.bfloat16)
Wa=torch.randn(C,2*C,device=dev,dtype=torch.bfloat16)
W0=torch.randn(C,C,device=dev,dtype=torch.bfloat16)
odd=g.as_strided((M,C),(2*C,1),C)
t=bench(lambda: (F.linear(g,Wa,bi), torch.mm(odd,W0.t())))
print(f"  Wa+W0 split    {t:.3f}ms {fl/t*1e3:.0f} TF")
t=bench(lambda: torch.mm(odd,W0.t())); print(f"    W0 strided-A  {t:.3f}ms")
oc=odd.contiguous()
t=bench(lambda: torch.mm(oc,W0.t())); print(f"    W0 contig-A   {t:.3f}ms")
del g,Wa,W0,odd,oc; torch.cuda.empty_cache()

print("=== aiter gemm ===")
x=torch.randn(M,C,device=dev,dtype=torch.bfloat16)
w=torch.randn(C,C,device=dev,dtype=torch.bfloat16)
flc=2*M*C*C/1e12
try:
    from aiter.ops.gemm_op_a16w16 import gemm_a16w16
    o=gemm_a16w16(x,w)
    t=bench(lambda: gemm_a16w16(x,w)); print(f"  aiter a16w16   {t:.3f}ms {flc/t*1e3:.0f} TF")
except Exception as e: print("  aiter gemm:",str(e)[:200])
t=bench(lambda: torch.mm(x,w.t())); print(f"  torch mm       {t:.3f}ms {flc/t*1e3:.0f} TF")
del x,w; torch.cuda.empty_cache()

print("=== elementwise costs ===")
h=torch.randn(B,T,C,device=dev,dtype=torch.bfloat16)
lw=torch.randn(C,device=dev,dtype=torch.bfloat16); lb=torch.randn(C,device=dev,dtype=torch.bfloat16)
t=bench(lambda: F.layer_norm(h,(C,),lw,lb,1e-5)); print(f"  layer_norm     {t:.3f}ms")
t=bench(lambda: h+h); print(f"  add            {t:.3f}ms")
e=torch.randn(T,C,device=dev,dtype=torch.bfloat16)
t=bench(lambda: h+e); print(f"  add bcast      {t:.3f}ms")
big=torch.randn(M,FF,device=dev,dtype=torch.bfloat16)
t=bench(lambda: F.gelu(big)); print(f"  gelu FF        {t:.3f}ms  ({4*M*FF/1e9:.1f}GB rw)")
t=bench(lambda: h.transpose(1,2).contiguous()); print(f"  permute BTC    {t:.3f}ms")
