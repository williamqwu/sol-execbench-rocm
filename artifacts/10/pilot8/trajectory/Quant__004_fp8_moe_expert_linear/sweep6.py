import torch, sys, os, time, triton, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2
torch.manual_seed(0)
def bench(f, n=50):
    try:
        for _ in range(10): f()
    except Exception as e: return None
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6
K=3584; NH=2048
res={}
for M in [384,640,896,1024,1536,2048,3072,4096]:
    aq=torch.randn(M,K,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    asc=torch.rand(M,K//128,device='cuda',dtype=torch.float32)+0.5
    w1q=torch.randn(2*NH,K,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w1s=torch.rand(2*NH//128,K//128,device='cuda',dtype=torch.float32)+0.5
    gq=torch.empty((M,NH),dtype=torch.float8_e4m3fn,device='cuda')
    gs=torch.empty((M,NH//128),dtype=torch.float32,device='cuda')
    best=[]
    for BM,BN,GM,nw,ns in itertools.product([16,32,64,128,256],[128,256],[1,4,8],[4,8],[1,2]):
        def f(BM=BM,BN=BN,GM=GM,nw=nw,ns=ns):
            v2.gemm1[(triton.cdiv(M,BM)*(NH//BN),)](aq,asc,w1q,w1s,gq,gs,M,K//128,NH,
              aq.stride(0),asc.stride(0),w1q.stride(0),w1s.stride(0),gq.stride(0),gs.stride(0),
              BLOCK_M=BM,BLOCK_N=BN,GROUP_M=GM,EVEN_M=(M%BM==0),num_warps=nw,num_stages=ns)
        t=bench(f)
        if t: best.append((t,BM,BN,GM,nw,ns))
    best.sort(); res[M]=best[0]
    print(f"M={M}: " + " | ".join(f"{t:.0f} BM{a}BN{b}GM{c}w{d}s{e}" for t,a,b,c,d,e in best[:4]), flush=True)
