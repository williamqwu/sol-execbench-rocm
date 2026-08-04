import torch, traceback
from tune import *

Bs = None
for M in [1, 53, 901]:
    A, B = mk(M)
    exp = torch.matmul(A, B.T)
    print(f"--- M={M} (torch gpu-only ref) ---")
    t = graph_time(lambda a,b: torch.matmul(a,b.T), (A,B))
    print(f"  torch                        {t:8.2f}us")
    best=[]
    # variant A: BM must be >=16 for tl.dot
    for BM in [16]:
        for BN in [16,32,64,128,256]:
            for BK in [64,128,256]:
                for ws in [4,8]:
                    for ns in [1,2,3]:
                        try:
                            f = make_A(BM,BN,BK,ws,ns)
                            o = f(A,B)
                            e = err(o,exp)
                            if e > 0.05: 
                                print(f"  A BM{BM} BN{BN} BK{BK} w{ws} s{ns}  ERR {e}")
                                continue
                            t = graph_time(f,(A,B))
                            best.append((t,f"A BM{BM} BN{BN} BK{BK} w{ws} s{ns}", e))
                        except Exception as ex:
                            pass
    best.sort()
    for t,n,e in best[:8]:
        print(f"  {n:34s} {t:8.2f}us  err={e:.5f}")
