import torch, triton
import tk2, reference
from chk import tol_ok

dev="cuda:0"; H=2048
torch.manual_seed(0)

def timeit(fn, iters=100, warmup=25):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Event(True); en=torch.cuda.Event(True)
    st.record()
    for _ in range(iters): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/iters*1000

SH=[(1,256),(4,128),(8,128),(4,512),(16,256),(64,128)]
CFGS=[(128,128,64,8,8,2),(128,128,64,8,8,3),(128,128,128,8,8,2),(128,256,64,8,8,2),
      (256,128,64,8,8,2),(64,128,64,8,4,2),(128,64,64,8,4,2),(64,64,64,8,4,2),
      (256,256,64,8,8,2),(128,128,32,8,8,2),(64,64,128,8,4,2),(64,128,128,8,4,2),
      (128,128,64,4,8,2),(128,128,64,1,8,2),(256,128,128,8,8,2),(128,256,128,8,8,2)]

for B,S in SH:
    go=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
    r=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
    w=torch.randn(H,H,device=dev,dtype=torch.bfloat16)
    g2=go.reshape(-1,H); r2=r.reshape(-1,H); M=B*S
    ra,rw=reference.run(go,r,w)
    tgw=timeit(lambda: g2.t()@r2)
    tga=timeit(lambda: (g2@w).view(B,S,32,64).permute(0,2,1,3).contiguous())
    bg=[];ba=[];bf=[]
    for c in CFGS:
        try:
            o=tk2.gw_run(g2,r2,c); q,_=tol_ok(o,rw,0.117)
            if q>=0.999: bg.append((timeit(lambda: tk2.gw_run(g2,r2,c)),c))
        except Exception: pass
        try:
            o=tk2.ga_run(g2,w,B,S,c); q,_=tol_ok(o,ra,0.117)
            if q>=0.999: ba.append((timeit(lambda: tk2.ga_run(g2,w,B,S,c)),c))
        except Exception: pass
        try:
            a,b=tk2.fused_run(g2,r2,w,B,S,c)
            q1,_=tol_ok(a,ra,0.117); q2,_=tol_ok(b,rw,0.117)
            if q1>=0.999 and q2>=0.999: bf.append((timeit(lambda: tk2.fused_run(g2,r2,w,B,S,c)),c))
        except Exception: pass
    bg.sort(); ba.sort(); bf.sort()
    print(f"M={M:5d} torch gw={tgw:6.1f} ga={tga:6.1f} sum={tgw+tga:6.1f}")
    print(f"        myGW {bg[0][0]:6.1f}{bg[0][1]}  myGA {ba[0][0]:6.1f}{ba[0][1]}  split={bg[0][0]+ba[0][0]:6.1f}")
    print(f"        fused {bf[0][0]:6.1f}{bf[0][1]} | {bf[1][0]:.1f}{bf[1][1]}", flush=True)
