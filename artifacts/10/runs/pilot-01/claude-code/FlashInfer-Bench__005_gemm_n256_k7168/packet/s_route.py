import torch, triton
from tune2 import *
# For M in the mid band, compare torch vs split-K vs tiled (gpu-only)
for M in [80,128,256,512,901,2048,4096]:
    A,B=mk(M) if M in _TOL else (torch.randn(M,7168,device=DEV,dtype=torch.float16),torch.randn(256,7168,device=DEV,dtype=torch.float16))
    exp=torch.matmul(A,B.T)
    tg=graph_time(lambda a,b: torch.matmul(a,b.T),(A,B))
    def chk(o):
        d=(o.float()-exp.float()).abs()
        return (d <= (0.044+0.001*exp.float().abs())).float().mean().item()>=0.99
    best=[]
    for BM in [16,32,64,128]:
        for BN in [32,64,128,256]:
            for SK in [1,2,4,8]:
                KS=7168//SK
                for BK in [64,128]:
                    if KS%BK: continue
                    for ws in [4,8]:
                        nb=triton.cdiv(M,BM)*(256//BN)*SK
                        if nb>16384: continue
                        try:
                            f=make_gsk(BM,BN,BK,SK,ws,2) if SK>1 else make_tiled(BM,BN,BK,ws,2)
                            o=f(A,B)
                            if not chk(o): continue
                            best.append((graph_time(f,(A,B)),f"{'gsk' if SK>1 else 'til'} BM{BM} BN{BN} BK{BK} SK{SK} w{ws} nb{nb}"))
                        except Exception: pass
    best.sort()
    print(f"M={M:5d} torch={tg:7.2f}  best: "+(f"{best[0][0]:7.2f} {best[0][1]}" if best else "none"),flush=True)
    for t,n in best[1:4]: print(f"            {t:7.2f} {n}",flush=True)
