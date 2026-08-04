import torch, sys, os, time, triton, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2, impl
torch.manual_seed(0)
def bench(f, n=50):
    try:
        for _ in range(10): f()
    except Exception: return None
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6
NH=2048; H=3584; K=3584
print("--- gemm2")
for M in [384,640,896,1024,1536,2048,3072,4096]:
    gq=torch.randn(M,NH,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    gs=torch.rand(M,NH//128,device='cuda',dtype=torch.float32)+0.5
    w2q=torch.randn(H,NH,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w2s=torch.rand(H//128,NH//128,device='cuda',dtype=torch.float32)+0.5
    rw=torch.randn(M,1,device='cuda',dtype=torch.bfloat16)
    out=torch.empty((M,H),dtype=torch.bfloat16,device='cuda')
    best=[]
    for BM,BN,GM,nw,ns in itertools.product([32,64,128,256],[128,256],[1,4],[4,8],[2]):
        def f(BM=BM,BN=BN,GM=GM,nw=nw,ns=ns):
            v2.gemm2[(triton.cdiv(M,BM)*(H//BN),)](gq,gs,w2q,w2s,rw,out,M,NH//128,H,
              gq.stride(0),gs.stride(0),w2q.stride(0),w2s.stride(0),out.stride(0),
              BLOCK_M=BM,BLOCK_N=BN,GROUP_M=GM,EVEN_M=(M%BM==0),num_warps=nw,num_stages=ns)
        t=bench(f)
        if t: best.append((t,BM,BN,GM,nw,ns))
    best.sort()
    print(f"M={M}: "+" | ".join(f"{t:.0f} BM{a}BN{b}GM{c}w{d}" for t,a,b,c,d,e in best[:4]), flush=True)
print("--- quant_act", flush=True)
for M in [384,1024,4096]:
    x=torch.randn(M,K,device='cuda',dtype=torch.bfloat16)
    q=torch.empty(M,K,device='cuda',dtype=torch.float8_e4m3fn)
    s=torch.empty(M,K//128,device='cuda',dtype=torch.float32)
    best=[]
    for BM,nw,ns in itertools.product([8,16,32,64,128],[1,2,4,8],[1,2]):
        def f(BM=BM,nw=nw,ns=ns):
            impl._quant_act_1x128[(triton.cdiv(M,BM),K//128)](x,q,s,M,x.stride(0),q.stride(0),s.stride(0),BLOCK_M=BM,num_warps=nw,num_stages=ns)
        t=bench(f,100)
        if t: best.append((t,BM,nw,ns))
    best.sort()
    print(f"M={M}: "+" | ".join(f"{t:.1f} BM{a}w{b}s{c}" for t,a,b,c in best[:4]), flush=True)
