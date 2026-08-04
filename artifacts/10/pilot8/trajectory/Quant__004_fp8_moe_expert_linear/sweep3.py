import torch, sys, os, time, triton, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triton.language as tl
FP8_MAX = tl.constexpr(448.0); FP8 = tl.constexpr(tl.float8e4nv)

@triton.jit
def qw(W, Q, S, KB: tl.constexpr, stride_wn, stride_qn, stride_sn,
       BK: tl.constexpr, BM: tl.constexpr):
    """each program: 128 rows x BK*? ; loop over sub-tiles"""
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    rn = tl.arange(0, BM)
    rk = tl.arange(0, BK)
    amax = tl.zeros((), dtype=tl.float32)
    base = W + (pid_n*128 + rn)[:,None]*stride_wn + (pid_k*128+rk)[None,:]
    for i in tl.static_range(128//BM):
        for j in tl.static_range(128//BK):
            w = tl.load(base + i*BM*stride_wn + j*BK).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(w)))
    scale = tl.maximum(amax * (1.0/FP8_MAX), 1e-12)
    inv = 1.0/scale
    qbase = Q + (pid_n*128 + rn)[:,None]*stride_qn + (pid_k*128+rk)[None,:]
    for i in tl.static_range(128//BM):
        for j in tl.static_range(128//BK):
            w = tl.load(base + i*BM*stride_wn + j*BK).to(tl.float32)
            q = tl.minimum(tl.maximum(w*inv, -FP8_MAX), FP8_MAX)
            tl.store(qbase + i*BM*stride_qn + j*BK, q.to(FP8))
    tl.store(S + pid_n*stride_sn + pid_k, scale)

def bench(f, n=100):
    try:
        for _ in range(20): f()
    except Exception as e: return None
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6

import impl
for N,K in [(4096,3584),(3584,2048)]:
    w=torch.randn(N,K,device='cuda',dtype=torch.bfloat16)
    q=torch.empty(N,K,device='cuda',dtype=torch.float8_e4m3fn)
    s=torch.empty(N//128,K//128,device='cuda',dtype=torch.float32)
    base=bench(lambda: impl.quantize_w(w))
    best=[]
    for BK,BM,nw,ns in itertools.product([32,64,128],[16,32,64,128],[1,2,4,8],[1,2]):
        def f(BK=BK,BM=BM,nw=nw,ns=ns):
            qw[(N//128,K//128)](w,q,s,K//128,w.stride(0),q.stride(0),s.stride(0),BK=BK,BM=BM,num_warps=nw,num_stages=ns)
        t=bench(f)
        if t: best.append((t,BK,BM,nw,ns))
    best.sort()
    print(f"N={N} K={K} base={base:.1f}us: " + " | ".join(f"{t:.1f} BK{a} BM{b} w{c} s{d}" for t,a,b,c,d in best[:5]))
