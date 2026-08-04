import torch, time, itertools
import torch.nn.functional as F
import triton, triton.language as tl
dev='cuda:0'
def bench(f, n=10):
    for _ in range(3): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3

B=16; T=1500; C=5120; FF=20480; H=20; D=256
M=B*T

@triton.jit
def _gemm(A,Bp,Cp, M,N,K, sam,sak, sbn,sbk, scm,scn,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid=tl.program_id(0)
    nm=tl.cdiv(M,BM); nn=tl.cdiv(N,BN)
    ng=GM*nn
    gid=pid//ng; fm=gid*GM; gs=min(nm-fm,GM)
    pm=fm+((pid%ng)%gs); pn=(pid%ng)//gs
    ram=(pm*BM+tl.arange(0,BM))%M
    rbn=(pn*BN+tl.arange(0,BN))%N
    rk=tl.arange(0,BK)
    a=A+(ram[:,None]*sam+rk[None,:]*sak)
    b=Bp+(rbn[None,:]*sbn+rk[:,None]*sbk)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for kk in range(0,tl.cdiv(K,BK)):
        av=tl.load(a); bv=tl.load(b)
        acc=tl.dot(av,bv,acc)
        a+=BK*sak; b+=BK*sbk
    o=acc.to(tl.bfloat16)
    rm=pm*BM+tl.arange(0,BM); rn=pn*BN+tl.arange(0,BN)
    tl.store(Cp+rm[:,None]*scm+rn[None,:]*scn, o, mask=(rm[:,None]<M)&(rn[None,:]<N))

def tgemm(a,b,BM,BN,BK,GM,ns,nw):
    M,K=a.shape; N=_=b.shape[0]
    c=torch.empty(M,N,device=a.device,dtype=torch.bfloat16)
    g=(triton.cdiv(M,BM)*triton.cdiv(N,BN),)
    _gemm[g](a,b,c,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),
             c.stride(0),c.stride(1),BM=BM,BN=BN,BK=BK,GM=GM,num_stages=ns,num_warps=nw)
    return c

for (name,m,n,k) in [("out_proj",M,C,C),("fc2",M,C,FF),("conv2",M,C,3*C)]:
    a=torch.randn(m,k,device=dev,dtype=torch.bfloat16)
    w=torch.randn(n,k,device=dev,dtype=torch.bfloat16)
    fl=2*m*n*k/1e12
    t=bench(lambda: torch.mm(a,w.t())); print(f"{name}: hipblaslt {t:.3f}ms {fl/t*1e3:.0f} TF")
    ref=torch.mm(a,w.t())
    best=(1e9,None)
    for BM,BN,BK,GM,ns,nw in itertools.product([128,256],[128,256],[64,128],[8],[2],[8]):
        try:
            o=tgemm(a,w,BM,BN,BK,GM,ns,nw)
            e=(o.float()-ref.float()).abs().max().item()
            t2=bench(lambda: tgemm(a,w,BM,BN,BK,GM,ns,nw),5)
            if t2<best[0]: best=(t2,(BM,BN,BK,GM,ns,nw,e))
        except Exception as ex: pass
    print(f"   triton best {best[0]:.3f}ms {fl/best[0]*1e3:.0f} TF cfg={best[1]}")
    del a,w,ref; torch.cuda.empty_cache()
