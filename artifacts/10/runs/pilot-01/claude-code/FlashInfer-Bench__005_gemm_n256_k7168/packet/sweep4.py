import sys, torch
from tune2 import *
MS=[int(x) for x in sys.argv[1].split(',')]
for M in MS:
    A,B=mk(M); exp=torch.matmul(A,B.T)
    tg=graph_time(lambda a,b: torch.matmul(a,b.T),(A,B))
    te=event_time(lambda a,b: torch.matmul(a,b.T),(A,B))
    print(f"--- M={M} --- torch gpu={tg:.2f} event={te:.2f}us")
    res=[]
    # split-K variants
    for BM in [16,32,64]:
        if BM > max(16,M*2): continue
        for BN in [16,32,64,128,256]:
            for SK in [1,2,4,8,16,32]:
                KS=7168//SK
                nblk=triton.cdiv(M,BM)*(256//BN)*SK
                if nblk > 4096: continue
                for BK in [64,128,256]:
                    if KS % BK: continue
                    for ws in [1,2,4,8]:
                        try:
                            f=make_gsk(BM,BN,BK,SK,ws,2)
                            o=f(A,B); ok,r,e=passes(o,exp,M)
                            if not ok: continue
                            res.append((graph_time(f,(A,B)),f"gsk BM{BM} BN{BN} BK{BK} SK{SK} w{ws} g{nblk}"))
                        except Exception: pass
    # tiled
    for BM in [16,32,64,128,256]:
        for BN in [16,32,64,128,256]:
            for BK in [64,128,256]:
                for ws in [2,4,8]:
                    for ns in [1,2]:
                        try:
                            f=make_tiled(BM,BN,BK,ws,ns)
                            o=f(A,B); ok,r,e=passes(o,exp,M)
                            if not ok: continue
                            res.append((graph_time(f,(A,B)),f"til BM{BM} BN{BN} BK{BK} w{ws} s{ns}"))
                        except Exception: pass
    res.sort()
    for t,n in res[:8]: print(f"   {n:44s} {t:7.2f}us")
