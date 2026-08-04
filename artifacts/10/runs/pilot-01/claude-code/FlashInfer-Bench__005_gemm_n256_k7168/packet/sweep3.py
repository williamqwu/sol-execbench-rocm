import torch
from tune import *
import sys
MS=[int(x) for x in sys.argv[1].split(',')]
for M in MS:
    A,B=mk(M); exp=torch.matmul(A,B.T)
    tt=graph_time(lambda a,b: torch.matmul(a,b.T),(A,B))
    print(f"--- M={M} --- torch {tt:.2f}us")
    res=[]
    for BN in [16,32,64,128]:
        for SK in [1,2,4,8,16,32,64]:
            NB=256//BN
            KS=7168//SK
            for BK in [32,64,128,256]:
                if KS % BK: continue
                for ws in [1,2,4,8]:
                    for ns in [1,2]:
                        try:
                            f=make_B(16,BN,BK,SK,ws,ns)
                            o=f(A,B); ok,r,e=passes(o,exp,M)
                            if not ok: continue
                            t=graph_time(f,(A,B))
                            res.append((t,f"BN{BN} SK{SK} BK{BK} w{ws} s{ns} g{NB*SK}",e,r))
                        except Exception: pass
    res.sort()
    for t,n,e,r in res[:10]: print(f"   {n:36s} {t:7.2f}us max={e:.4f} m={r:.4f}")
