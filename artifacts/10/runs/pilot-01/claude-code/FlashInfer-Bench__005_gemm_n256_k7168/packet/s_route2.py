import torch, triton
from tune2 import *
def chk(o,exp):
    d=(o.float()-exp.float()).abs()
    return (d <= (0.043+0.00097*exp.float().abs())).float().mean().item()>=0.99
for M in [53,63,80,901]:
    A,B=mk(M); exp=torch.matmul(A,B.T)
    tg=graph_time(lambda a,b: torch.matmul(a,b.T),(A,B))
    res=[]
    for BM in [16,32,64,128]:
        for BN in [16,32,64,128]:
            for SK in [1,2,4,8,16,32]:
                KS=7168//SK
                nb=triton.cdiv(M,BM)*(256//BN)*SK
                if nb>16384: continue
                for BK in [64,128]:
                    if KS%BK: continue
                    for ws in [1,2,4,8]:
                        try:
                            f=make_gsk(BM,BN,BK,SK,ws,2)
                            o=f(A,B)
                            if not chk(o,exp): continue
                            res.append((graph_time(f,(A,B)),f"gsk BM{BM} BN{BN} BK{BK} SK{SK} w{ws} nb{nb}"))
                        except Exception: pass
    res.sort()
    print(f"M={M:5d} torch={tg:7.2f}",flush=True)
    for t,n in res[:5]: print(f"        {t:7.2f} {n}",flush=True)
