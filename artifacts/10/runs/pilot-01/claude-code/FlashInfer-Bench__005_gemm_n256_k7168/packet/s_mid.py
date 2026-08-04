import torch, triton
from tune2 import *
# mid/small M: split-K vs tiled
for M in [4,16,53,63,80,901]:
    A,B=mk(M); exp=torch.matmul(A,B.T)
    tg=graph_time(lambda a,b: torch.matmul(a,b.T),(A,B))
    print(f"--- M={M} --- torch gpu={tg:.2f}us",flush=True)
    res=[]
    for BM in [16,32,64,128]:
        for BN in [16,32,64,128]:
            for SK in [1,2,4,8,16,32]:
                KS=7168//SK
                nb=triton.cdiv(M,BM)*(256//BN)*SK
                if nb>8192: continue
                for BK in [64,128,256]:
                    if KS%BK: continue
                    for ws in [1,2,4,8]:
                        try:
                            f=make_gsk(BM,BN,BK,SK,ws,2)
                            o=f(A,B); ok,r,e=passes(o,exp,M)
                            if not ok: continue
                            res.append((graph_time(f,(A,B)),f"gsk BM{BM} BN{BN} BK{BK} SK{SK} w{ws} nb{nb}"))
                        except Exception: pass
    res.sort()
    for t,n in res[:6]: print(f"   {n:46s} {t:7.2f}us",flush=True)
