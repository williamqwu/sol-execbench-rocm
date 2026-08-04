import torch, sys, os, time, triton, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impl
torch.manual_seed(0)
def bench(f, n=50):
    try:
        for _ in range(10): f()
    except Exception: return None
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6
NH=2048; H=3584
for M in [384, 1024, 2048, 4096]:
    gq=torch.randn(M,NH,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    gs=torch.rand(M,NH//128,device='cuda',dtype=torch.float32)+0.5
    w2q=torch.randn(H,NH,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w2s=torch.rand(H//128,NH//128,device='cuda',dtype=torch.float32)+0.5
    rw=torch.randn(M,1,device='cuda',dtype=torch.bfloat16)
    out=torch.empty((M,H),dtype=torch.bfloat16,device='cuda')
    best=[]
    for BM, BN, GM, nw, ns in itertools.product([32,64,128,256],[32,64,128],[1,4,8],[4,8],[1,2,3]):
        def g2(BM=BM,BN=BN,GM=GM,nw=nw,ns=ns):
            impl._gemm2[(triton.cdiv(M,BM)*triton.cdiv(H,BN),)](gq,gs,w2q,w2s,rw,out,M,NH//128,H//128,
              gq.stride(0),gs.stride(0),w2q.stride(0),w2s.stride(0),out.stride(0),
              BLOCK_M=BM,BLOCK_N=BN,GROUP_M=GM,EVEN_M=(M%BM==0),num_warps=nw,num_stages=ns)
        t=bench(g2)
        if t: best.append((t,BM,BN,GM,nw,ns))
    best.sort()
    print(f"M={M}: " + " | ".join(f"{t:.0f}us BM{b} BN{n} GM{g} w{w} s{s}" for t,b,n,g,w,s in best[:5]))
