import torch, triton, itertools
import tk, tf, tf2, reference
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
C1=[(128,128,64,8),(128,256,64,4),(256,128,64,4),(128,128,128,8),(256,256,64,2),(64,128,64,8),(128,64,64,8)]
C2=[(128,128,64,8),(128,256,64,4),(256,128,64,4),(64,128,64,8),(64,64,64,8),(128,64,64,8),(256,256,64,2),(32,128,64,8)]
NWNS=[(8,2),(8,3),(4,2)]

for B,S in SH:
    go=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
    r=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
    w=torch.randn(H,H,device=dev,dtype=torch.bfloat16)
    g2=go.reshape(-1,H); r2=r.reshape(-1,H); M=B*S
    ra,rw = reference.run(go,r,w)
    def sep():
        a=tk.ga_run(g2,w,B,S,(128,128,64,8,8,2))
        b=tk.gw_run(g2,r2,(128,128,64,8,8,3))
        return a,b
    ts=timeit(sep)
    best=[]
    for c1 in C1:
        for c2 in C2:
            for nw,ns in NWNS:
                try:
                    ga,gw=tf2.run2(g2,r2,w,B,S,c1,c2,nw,ns)
                    q1,_=tol_ok(ga,ra,0.117); q2,_=tol_ok(gw,rw,0.117)
                    if q1<0.999 or q2<0.999: continue
                    t=timeit(lambda: tf2.run2(g2,r2,w,B,S,c1,c2,nw,ns))
                    best.append((t,c1,c2,nw,ns))
                except Exception as e:
                    pass
    best.sort()
    print(f"M={M:5d} sep={ts:7.1f} | " + " | ".join(f"{t:.1f} {c1}{c2}w{nw}s{ns}" for t,c1,c2,nw,ns in best[:3]), flush=True)
