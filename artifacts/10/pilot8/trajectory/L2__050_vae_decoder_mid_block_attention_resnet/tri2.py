import torch, triton, triton.language as tl, time, itertools
dev='cuda'
@triton.jit
def gk(A,B,Cc,M,N,K,sam,sbn,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SPLIT:tl.constexpr,GM:tl.constexpr):
    pid=tl.program_id(0)
    nm=tl.cdiv(M,BM); nn=tl.cdiv(N,BN)
    ng=GM*nn; gid=pid//ng; fm=gid*GM; gs=min(nm-fm,GM)
    pid_m=fm+((pid%ng)%gs); pid_n=(pid%ng)//gs
    rm=(pid_m*BM+tl.arange(0,BM))%M; rn=(pid_n*BN+tl.arange(0,BN))%N
    rk=tl.arange(0,BK)
    ap=A+rm[:,None]*sam+rk[None,:]
    bp=B+rn[:,None]*sbn+rk[None,:]   # B is NxK (transposed weights)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    acl=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        a=tl.load(ap); b=tl.load(bp)
        ah=a.to(tl.float16); bh=b.to(tl.float16)
        acc=tl.dot(ah,tl.trans(bh),acc)
        if SPLIT:
            am=((a-ah.to(tl.float32))*4096.0).to(tl.float16)
            bm=((b-bh.to(tl.float32))*4096.0).to(tl.float16)
            acl=tl.dot(ah,tl.trans(bm),acl)
            acl=tl.dot(am,tl.trans(bh),acl)
        ap+=BK; bp+=BK
    o=acc+acl*(1.0/4096.0)
    rm2=pid_m*BM+tl.arange(0,BM); rn2=pid_n*BN+tl.arange(0,BN)
    tl.store(Cc+rm2[:,None]*N+rn2[None,:], o, mask=(rm2[:,None]<M)&(rn2[None,:]<N))

def bench(f,n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000

M,K,N=32768,4608,512
a=torch.randn(M,K,device=dev); bT=torch.randn(N,K,device=dev)
ref=a.double()@bT.double().t()
tf32=bench(lambda: a@bT.t()); print(f"rocblas fp32 {tf32:.3f}ms {2*M*N*K/tf32*1e-9:.0f} TF")
best=None
for BM,BN,BK,nw,ns,gm in itertools.product([128,256],[64,128,256],[32,64],[4,8],[1,2],[4,8]):
    if BM*BN>256*128: continue
    c=torch.empty(M,N,device=dev)
    try:
        f=lambda: gk[(triton.cdiv(M,BM)*triton.cdiv(N,BN),)](a,bT,c,M,N,K,K,K,BM,BN,BK,True,gm,num_warps=nw,num_stages=ns)
        t=bench(f,10)
    except Exception as e:
        continue
    e=((c.double()-ref).abs()/ref.abs().clamp(min=1e-2)).median().item()
    if best is None or t<best[0]: best=(t,BM,BN,BK,nw,ns,gm,e)
    print(f"BM{BM} BN{BN} BK{BK} w{nw} s{ns} g{gm}: {t:.3f} {2*M*N*K/t*1e-9:.0f}TF err{e:.1e}")
print("BEST",best)
