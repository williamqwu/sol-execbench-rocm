import torch, time, triton, triton.language as tl, itertools
@triton.jit
def pure(A,B,C,M,K:tl.constexpr,NB:tl.constexpr,sa,sb,sc,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pid=tl.program_id(0); npm=tl.cdiv(M,BM)
    pm=pid%npm; pn=pid//npm
    rm=pm*BM+tl.arange(0,BM); rn=pn*BN+tl.arange(0,BN); rk=tl.arange(0,BK)
    ap=A+rm[:,None]*sa+rk[None,:]; bp=B+rn[:,None]*sb+rk[None,:]
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k in range(0,K//BK):
        acc=tl.dot(tl.load(ap),tl.trans(tl.load(bp)),acc)
        ap+=BK; bp+=BK
    tl.store(C+rm[:,None]*sc+rn[None,:],acc.to(tl.bfloat16))
def bench(f,n=50):
    try:
        for _ in range(10): f()
    except Exception: return None
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6
for M,N,K in [(4096,4096,3584),(4096,3584,2048),(384,4096,3584)]:
    a=torch.randn(M,K,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b=torch.randn(N,K,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    c=torch.empty(M,N,device='cuda',dtype=torch.bfloat16)
    best=[]
    for BM,BN,BK,nw,ns in itertools.product([64,128,256],[64,128,256],[64,128,256],[4,8],[1,2,3]):
        if M%BM or N%BN: continue
        def f(BM=BM,BN=BN,BK=BK,nw=nw,ns=ns):
            pure[(M//BM*(N//BN),)](a,b,c,M,K,N//BN,a.stride(0),b.stride(0),c.stride(0),BM=BM,BN=BN,BK=BK,num_warps=nw,num_stages=ns)
        t=bench(f)
        if t: best.append((t,BM,BN,BK,nw,ns))
    best.sort()
    fl=2*M*N*K
    print(f"{M}x{N}x{K}: "+" | ".join(f"{t:.0f}us({fl/t/1e6:.0f}TF) {a_}/{b_}/{c_} w{d} s{e}" for t,a_,b_,c_,d,e in best[:4]))
