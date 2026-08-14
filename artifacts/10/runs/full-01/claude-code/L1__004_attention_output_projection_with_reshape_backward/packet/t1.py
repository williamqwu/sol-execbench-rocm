import torch, triton, itertools, sys
import tk

dev="cuda:0"; H=2048
torch.manual_seed(0)

def timeit(fn, iters=100, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Event(True); en=torch.cuda.Event(True)
    st.record()
    for _ in range(iters): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/iters*1000

SH=[(1,256),(4,128),(8,128),(4,512),(16,256),(64,128),(2,4096)]

def mk(B,S):
    go=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
    r=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
    w=torch.randn(H,H,device=dev,dtype=torch.bfloat16)
    return go,r,w

GW_CFGS=[(128,128,64,8,8,2),(128,128,128,8,8,2),(128,128,64,4,8,2),
         (256,128,64,8,8,2),(128,256,64,8,8,2),(256,256,64,8,8,2),
         (128,128,32,8,4,2),(64,128,64,8,4,2),(128,64,64,8,4,2),
         (128,128,64,8,8,3),(256,128,128,8,8,2),(128,128,128,4,8,3)]
GA_CFGS=[(128,128,64,8,8,2),(128,256,64,8,8,2),(256,128,64,8,8,2),
         (64,128,64,8,4,2),(128,128,128,8,8,2),(64,256,64,8,4,2),
         (256,256,64,8,8,2),(32,128,64,8,4,2),(128,64,64,8,4,2),
         (64,64,64,8,4,2),(128,128,64,4,8,3),(64,128,128,8,4,2)]

print("=== GW (goT @ r -> 2048x2048) ===")
for B,S in SH:
    go,r,w=mk(B,S); M=B*S
    g2=go.reshape(-1,H); r2=r.reshape(-1,H)
    ref=(g2.t()@r2)
    tt=timeit(lambda: g2.t()@r2)
    best=[]
    for cfg in GW_CFGS:
        try:
            out=tk.gw_run(g2,r2,cfg)
            err=(out.float()-ref.float()).abs().max().item()
            t=timeit(lambda: tk.gw_run(g2,r2,cfg))
            best.append((t,cfg,err))
        except Exception as e:
            pass
    best.sort()
    print(f" M={M:5d} torch={tt:8.1f} " + " | ".join(f"{t:.1f}{cfg}e{e:.2f}" for t,cfg,e in best[:3]))

print("=== GA (go @ w -> permuted) ===")
for B,S in SH:
    go,r,w=mk(B,S); M=B*S
    g2=go.reshape(-1,H)
    ref=(g2@w).view(B,S,32,64).permute(0,2,1,3).contiguous()
    tt=timeit(lambda: (g2@w).view(B,S,32,64).permute(0,2,1,3).contiguous())
    best=[]
    for cfg in GA_CFGS:
        try:
            out=tk.ga_run(g2,w,B,S,cfg)
            err=(out.float()-ref.float()).abs().max().item()
            t=timeit(lambda: tk.ga_run(g2,w,B,S,cfg))
            best.append((t,cfg,err))
        except Exception as e:
            pass
    best.sort()
    print(f" M={M:5d} torch={tt:8.1f} " + " | ".join(f"{t:.1f}{cfg}e{e:.2f}" for t,cfg,e in best[:3]))
