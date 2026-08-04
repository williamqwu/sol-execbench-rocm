import torch, sys, os, time, triton, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triton.language as tl
FP8_MAX = tl.constexpr(448.0); FP8 = tl.constexpr(tl.float8e4nv)

# one program per (128-row block, 128-col block); rows split across NSPLIT waves via
# 2D tile of (RM x 128). amax reduced with atomic-free two-pass in registers.
@triton.jit
def qw2(W, Q, S, stride_wn, stride_qn, stride_sn, RM: tl.constexpr):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    rk = tl.arange(0, 128)
    rm = tl.arange(0, RM)
    off = (pid_n*128 + rm)[:,None]*stride_wn + (pid_k*128+rk)[None,:]
    amax = tl.zeros((), dtype=tl.float32)
    for i in tl.static_range(128//RM):
        w = tl.load(W + off + i*RM*stride_wn)
        amax = tl.maximum(amax, tl.max(tl.abs(w.to(tl.float32))))
    scale = tl.maximum(amax*(1.0/FP8_MAX), 1e-12); inv = 1.0/scale
    offq = (pid_n*128 + rm)[:,None]*stride_qn + (pid_k*128+rk)[None,:]
    for i in tl.static_range(128//RM):
        w = tl.load(W + off + i*RM*stride_wn).to(tl.float32)
        tl.store(Q + offq + i*RM*stride_qn, tl.minimum(tl.maximum(w*inv,-FP8_MAX),FP8_MAX).to(FP8))
    tl.store(S + pid_n*stride_sn + pid_k, scale)

# variant: full 128x128 in registers, single load
@triton.jit
def qw3(W, Q, S, stride_wn, stride_qn, stride_sn):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    rk = tl.arange(0,128); rn = tl.arange(0,128)
    w = tl.load(W + (pid_n*128+rn)[:,None]*stride_wn + (pid_k*128+rk)[None,:])
    wf = w.to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(wf))*(1.0/FP8_MAX), 1e-12)
    tl.store(Q + (pid_n*128+rn)[:,None]*stride_qn + (pid_k*128+rk)[None,:],
             tl.minimum(tl.maximum(wf*(1.0/scale),-FP8_MAX),FP8_MAX).to(FP8))
    tl.store(S + pid_n*stride_sn + pid_k, scale)

def bench(f, n=200):
    try:
        for _ in range(20): f()
    except Exception: return None
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6

for N,K in [(4096,3584),(3584,2048)]:
    w=torch.randn(N,K,device='cuda',dtype=torch.bfloat16)
    q=torch.empty(N,K,device='cuda',dtype=torch.float8_e4m3fn)
    s=torch.empty(N//128,K//128,device='cuda',dtype=torch.float32)
    tc = bench(lambda: w.to(torch.float8_e4m3fn))
    best=[]
    for RM,nw,ns in itertools.product([8,16,32,64,128],[1,2,4,8,16],[1,2]):
        def f(RM=RM,nw=nw,ns=ns): qw2[(N//128,K//128)](w,q,s,w.stride(0),q.stride(0),s.stride(0),RM=RM,num_warps=nw,num_stages=ns)
        t=bench(f)
        if t: best.append((t,'RM%d w%d s%d'%(RM,nw,ns)))
    for nw,ns in itertools.product([1,2,4,8,16],[1,2]):
        def f(nw=nw,ns=ns): qw3[(N//128,K//128)](w,q,s,w.stride(0),q.stride(0),s.stride(0),num_warps=nw,num_stages=ns)
        t=bench(f)
        if t: best.append((t,'full w%d s%d'%(nw,ns)))
    best.sort()
    print(f"N={N} K={K} torch_cast={tc:.1f}us: " + " | ".join(f"{t:.1f} {d}" for t,d in best[:6]))
