import sys, torch, triton
from tune2 import *
for M in [14104, 11948, 901]:
    A,B=mk(M); exp=torch.matmul(A,B.T)
    tg=graph_time(lambda a,b: torch.matmul(a,b.T),(A,B))
    byt=M*7168*2+256*7168*2+M*256*2
    print(f"--- M={M} --- torch gpu={tg:.2f}us  ({byt/(tg*1e-6)/1e9:.0f} GB/s)",flush=True)
    res=[]
    for BM in [32,64,128,256]:
        for BN in [32,64,128,256]:
            for BK in [32,64,128]:
                for ws in [4,8]:
                    for ns in [1,2,3]:
                        nb=triton.cdiv(M,BM)*(256//BN)
                        try:
                            f=make_tiled(BM,BN,BK,ws,ns)
                            o=f(A,B); ok,r,e=passes(o,exp,M)
                            if not ok: continue
                            res.append((graph_time(f,(A,B)),f"til BM{BM} BN{BN} BK{BK} w{ws} s{ns} nb{nb}"))
                        except Exception: pass
    res.sort()
    for t,n in res[:10]:
        print(f"   {n:42s} {t:7.2f}us  {byt/(t*1e-6)/1e9:.0f} GB/s",flush=True)
