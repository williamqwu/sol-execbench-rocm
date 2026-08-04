import torch
from tune import *

for M in [1, 16]:
    A, B = mk(M)
    exp = torch.matmul(A, B.T)
    print(f"--- M={M} --- torch {graph_time(lambda a,b: torch.matmul(a,b.T),(A,B)):.2f}us")
    res=[]
    for BN in [16,32,64]:
        for SK in [4,8,16,32,64]:
            NB = 256//BN
            if NB*SK > 2048: continue
            KS = 7168//SK
            for BK in [32,64,128,256]:
                if KS % BK: continue
                for ws in [1,2,4,8]:
                    for ns in [1,2]:
                        try:
                            f = make_B(16,BN,BK,SK,ws,ns)
                            o=f(A,B); e=err(o,exp)
                            if e>0.05: continue
                            t=graph_time(f,(A,B))
                            res.append((t,f"BN{BN} SK{SK} BK{BK} w{ws} s{ns} grid{NB*SK}",e))
                        except Exception as ex: pass
    res.sort()
    for t,n,e in res[:10]: print(f"   {n:38s} {t:7.2f}us err={e:.5f}")
