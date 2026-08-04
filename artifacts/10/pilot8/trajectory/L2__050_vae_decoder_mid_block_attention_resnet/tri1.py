import torch, triton, triton.language as tl, time
dev='cuda'

@triton.jit
def gemm3(A,B,Cc, M,N,K, sam,sak, sbk,sbn, scm,scn,
          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr, SPLIT:tl.constexpr):
    pid_m=tl.program_id(0); pid_n=tl.program_id(1)
    rm=pid_m*BM+tl.arange(0,BM); rn=pid_n*BN+tl.arange(0,BN)
    rk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    acl=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        a=tl.load(A+rm[:,None]*sam+(k0+rk)[None,:]*sak)
        b=tl.load(B+(k0+rk)[:,None]*sbk+rn[None,:]*sbn)
        ah=a.to(tl.float16); bh=b.to(tl.float16)
        acc=tl.dot(ah,bh,acc)
        if SPLIT:
            am=((a-ah.to(tl.float32))*2048.0).to(tl.float16)
            bm=((b-bh.to(tl.float32))*2048.0).to(tl.float16)
            acl=tl.dot(ah,bm,acl)
            acl=tl.dot(am,bh,acl)
    o=acc+acl*(1.0/2048.0)
    tl.store(Cc+rm[:,None]*scm+rn[None,:]*scn, o)

def run(a,b,split=True,BM=128,BN=128,BK=64):
    M,K=a.shape; K2,N=b.shape
    c=torch.empty(M,N,device=dev,dtype=torch.float32)
    gemm3[(M//BM,N//BN)](a,b,c,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),c.stride(0),c.stride(1),BM,BN,BK,split,num_warps=8,num_stages=2)
    return c

def bench(f,n=20):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000

M,K,N=4096,4608,512
a=torch.randn(M,K,device=dev); b=torch.randn(K,N,device=dev)
ref=(a.double()@b.double())
f32=a@b
def err(r): return ((r.double()-ref).abs()/ref.abs().clamp(min=1e-2)).median().item()
print("fp32 err", err(f32), "t", bench(lambda:a@b), "TF", 2*M*N*K/bench(lambda:a@b)*1e-9)
c=run(a,b,True); print("split3 err", err(c), "t",bench(lambda:run(a,b,True)), "TF",2*M*N*K/bench(lambda:run(a,b,True))*1e-9)
c=run(a,b,False); print("fp16 err", err(c), "t",bench(lambda:run(a,b,False)))
