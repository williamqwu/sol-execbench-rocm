import torch, triton
import tk, tf, reference
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

SH=[(1,256),(4,128),(8,128),(4,512),(16,256),(64,128)]
CFGS=[(128,128,64,8,8,2),(128,128,64,8,8,3),(128,128,128,8,8,2),
      (128,64,64,8,4,2),(64,128,64,8,4,2),(64,64,64,8,4,2),
      (128,256,64,8,8,2),(256,128,64,8,8,2),(64,128,128,8,4,2),
      (128,128,32,8,8,2),(64,64,128,8,4,2)]

for B,S in SH:
    go=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
    r=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
    w=torch.randn(H,H,device=dev,dtype=torch.bfloat16)
    g2=go.reshape(-1,H); r2=r.reshape(-1,H); M=B*S
    ra,rw = reference.run(go,r,w)
    # separate-kernel baseline (best cfgs found)
    def sep():
        a=tk.ga_run(g2,w,B,S,(128,128,64,8,8,2))
        b=tk.gw_run(g2,r2,(128,128,64,8,8,3))
        return a,b
    ts=timeit(sep)
    best=[]
    for cfg in CFGS:
        try:
            ga,gw=tf.fused_run(g2,r2,w,B,S,cfg)
            r1,_=tol_ok(ga,ra,0.117); r2c,_=tol_ok(gw,rw,0.117)
            if r1<0.999 or r2c<0.999:
                print("  BAD",cfg,r1,r2c); continue
            t=timeit(lambda: tf.fused_run(g2,r2,w,B,S,cfg))
            best.append((t,cfg))
        except Exception as e:
            print("  EX",cfg,str(e)[:60])
    best.sort()
    print(f"M={M:5d} sep={ts:7.1f} fused: " + " | ".join(f"{t:.1f}{cfg}" for t,cfg in best[:4]))
