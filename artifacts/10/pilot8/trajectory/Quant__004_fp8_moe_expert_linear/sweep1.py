import torch, sys, os, time, triton, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference, impl
torch.manual_seed(0)
def bench(f, n=50):
    try:
        for _ in range(10): f()
    except Exception as e:
        return None
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6
K=3584; NH=2048; H=3584
for M in [384, 1024, 2048, 4096]:
    aq=torch.randn(M,K,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    asc=torch.rand(M,K//128,device='cuda',dtype=torch.float32)+0.5
    w1q=torch.randn(2*NH,K,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w1s=torch.rand(2*NH//128,K//128,device='cuda',dtype=torch.float32)+0.5
    gq=torch.empty((M,NH),dtype=torch.float8_e4m3fn,device='cuda')
    gs=torch.empty((M,NH//128),dtype=torch.float32,device='cuda')
    best=[]
    for BM, GM, nw, ns in itertools.product([32,64,128,256],[1,4,8],[4,8],[1,2,3]):
        if BM*256*4*2 // (nw*64) > 512*4: continue
        def g1(BM=BM,GM=GM,nw=nw,ns=ns):
            impl._gemm1_silu_quant[(triton.cdiv(M,BM)*(NH//128),)](aq,asc,w1q,w1s,gq,gs,M,K//128,NH//128,
              aq.stride(0),asc.stride(0),w1q.stride(0),w1s.stride(0),gq.stride(0),gs.stride(0),
              BLOCK_M=BM,GROUP_M=GM,EVEN_M=(M%BM==0),num_warps=nw,num_stages=ns)
        t=bench(g1)
        if t: best.append((t,BM,GM,nw,ns))
    best.sort()
    print(f"M={M}: " + " | ".join(f"{t:.0f}us BM{b} GM{g} w{w} s{s}" for t,b,g,w,s in best[:5]))
