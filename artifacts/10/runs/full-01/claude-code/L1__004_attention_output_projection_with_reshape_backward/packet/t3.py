import torch, triton
import tk
from chk import tol_ok

dev="cuda:0"; H=2048
torch.manual_seed(0)

def timeit(fn, iters=200, warmup=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Event(True); en=torch.cuda.Event(True)
    st.record()
    for _ in range(iters): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/iters*1000

import reference

SH=[(1,256),(4,128),(8,128),(4,512),(16,256),(64,128)]

GA_CFGS=[(128,128,64,8,8,2),(128,256,64,8,8,2),(256,128,64,8,8,2),
         (64,128,64,8,4,2),(128,128,128,8,8,2),(64,256,64,8,4,2),
         (32,128,64,8,4,2),(128,64,64,8,4,2),(64,64,64,8,4,2),
         (32,64,64,8,4,2),(32,256,64,8,4,2),(16,128,64,8,4,2),
         (64,128,128,8,4,2),(32,128,128,8,4,2),(128,128,64,8,8,3),
         (256,256,64,8,8,2),(64,128,64,8,8,2)]

print("=== GA vs true fp32 reference ===")
for B,S in SH:
    go=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
    r=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
    w=torch.randn(H,H,device=dev,dtype=torch.bfloat16)
    g2=go.reshape(-1,H); M=B*S
    ra, rw = reference.run(go,r,w)
    tt=timeit(lambda: (g2@w).view(B,S,32,64).permute(0,2,1,3).contiguous())
    tb=(g2@w).view(B,S,32,64).permute(0,2,1,3).contiguous()
    mr,mx=tol_ok(tb,ra,0.117)
    best=[]
    for cfg in GA_CFGS:
        try:
            out=tk.ga_run(g2,w,B,S,cfg)
            r2,m2=tol_ok(out,ra,0.117)
            t=timeit(lambda: tk.ga_run(g2,w,B,S,cfg))
            best.append((t,cfg,r2,m2))
        except Exception as e:
            pass
    best.sort()
    print(f" M={M:5d} torch={tt:7.1f}(ratio{mr:.4f} mx{mx:.2f}) " +
          " | ".join(f"{t:.1f}{cfg}r{rr:.4f}" for t,cfg,rr,_ in best[:3]))
