import torch, sys, os, time, triton, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impl
torch.manual_seed(0)
def bench(f, n=50):
    try:
        for _ in range(10): f()
    except Exception as e: return None
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6
K=3584; NH=2048
for M,BM in [(384,64),(1024,64),(2048,128),(4096,128)]:
    aq=torch.randn(M,K,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    asc=torch.rand(M,K//128,device='cuda',dtype=torch.float32)+0.5
    w1q=torch.randn(2*NH,K,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w1s=torch.rand(2*NH//128,K//128,device='cuda',dtype=torch.float32)+0.5
    gq=torch.empty((M,NH),dtype=torch.float8_e4m3fn,device='cuda')
    gs=torch.empty((M,NH//128),dtype=torch.float32,device='cuda')
    best=[]
    for nd,kp,wpe,ns in itertools.product([16,32],[1,2],[0,1,2,4],[1,2,3]):
        kw = dict(matrix_instr_nonkdim=nd, kpack=kp, num_stages=ns, num_warps=8)
        if wpe: kw['waves_per_eu']=wpe
        def g1(kw=kw):
            impl._gemm1_silu_quant[(triton.cdiv(M,BM)*(NH//128),)](aq,asc,w1q,w1s,gq,gs,M,K//128,NH//128,
              aq.stride(0),asc.stride(0),w1q.stride(0),w1s.stride(0),gq.stride(0),gs.stride(0),
              BLOCK_M=BM,GROUP_M=1,EVEN_M=(M%BM==0),**kw)
        t=bench(g1)
        if t: best.append((t,nd,kp,wpe,ns))
    best.sort()
    print(f"M={M} BM={BM}: " + " | ".join(f"{t:.0f} nd{a} kp{b} wpe{c} s{d}" for t,a,b,c,d in best[:5]))
