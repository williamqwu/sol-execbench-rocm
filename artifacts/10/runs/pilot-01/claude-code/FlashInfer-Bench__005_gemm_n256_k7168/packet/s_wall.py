import time, torch, triton
from tune2 import *

def wall(fn,args,it=200):
    for _ in range(30): fn(*args)
    torch.cuda.synchronize()
    b=1e18
    for _ in range(5):
        t0=time.perf_counter()
        for _ in range(it): fn(*args)
        torch.cuda.synchronize()
        b=min(b,(time.perf_counter()-t0)/it*1e6)
    return b

# where does time go at M=1?
A,B=mk(1)
print("torch.matmul wall  %.2f"%wall(lambda a,b: torch.matmul(a,b.T),(A,B)),flush=True)
f2=make_gsk(16,16,64,16,2,2)
print("gsk 2-launch wall  %.2f  gpu %.2f"%(wall(f2,(A,B)),graph_time(f2,(A,B))),flush=True)

# single-launch tiled (no splitk) small M
for BN in [16,32]:
    for BK in [128,256]:
        for ws in [1,2,4]:
            f=make_tiled(16,BN,BK,ws,2)
            try:
                o=f(A,B)
                print(f"til BN{BN} BK{BK} w{ws}: wall %.2f gpu %.2f"%(wall(f,(A,B)),graph_time(f,(A,B))),flush=True)
            except Exception as e: print("err",e)
