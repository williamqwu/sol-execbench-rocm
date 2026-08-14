import torch, triton, itertools, json
import tk2, reference
from chk import tol_ok

dev="cuda:0"; H=2048
torch.manual_seed(0)

def timeit(fn, iters=150, warmup=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Event(True); en=torch.cuda.Event(True)
    st.record()
    for _ in range(iters): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/iters*1000

SH=[(1,256),(4,128),(8,128),(4,512),(16,256),(64,128)]

CFGS=[]
for BM in [64,128,256]:
    for BN in [64,128,256]:
        for BK in [32,64,128]:
            for G in [1,4,8]:
                nw = 8 if (BM*BN>=128*128) else 4
                for ns in [2,3]:
                    CFGS.append((BM,BN,BK,G,nw,ns))
print("ncfg",len(CFGS))

res={}
for B,S in SH:
    go=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
    r=torch.randn(B,S,H,device=dev,dtype=torch.bfloat16)
    w=torch.randn(H,H,device=dev,dtype=torch.bfloat16)
    g2=go.reshape(-1,H); r2=r.reshape(-1,H); M=B*S
    ra,rw=reference.run(go,r,w)
    tsum=timeit(lambda: g2.t()@r2)+timeit(lambda: (g2@w).view(B,S,32,64).permute(0,2,1,3).contiguous())
    bf=[]
    for c in CFGS:
        try:
            a,b=tk2.fused_run(g2,r2,w,B,S,c)
            q1,_=tol_ok(a,ra,0.117); q2,_=tol_ok(b,rw,0.117)
            if q1<0.999 or q2<0.999: continue
            bf.append((timeit(lambda: tk2.fused_run(g2,r2,w,B,S,c)),c))
        except Exception: pass
    bf.sort()
    res[M]=bf[:5]
    print(f"M={M:5d} torchsum={tsum:7.1f} best: " + " | ".join(f"{t:.1f}{c}" for t,c in bf[:5]), flush=True)

json.dump({str(k):[[t,list(c)] for t,c in v] for k,v in res.items()}, open("fused_sweep.json","w"), indent=1)
