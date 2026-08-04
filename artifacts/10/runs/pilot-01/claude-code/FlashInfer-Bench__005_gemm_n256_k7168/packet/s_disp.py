import time, torch, triton, triton.language as tl
from tune2 import *

def wall(fn,args,it=300):
    for _ in range(30): fn(*args)
    torch.cuda.synchronize()
    b=1e18
    for _ in range(5):
        t0=time.perf_counter()
        for _ in range(it): fn(*args)
        torch.cuda.synchronize()
        b=min(b,(time.perf_counter()-t0)/it*1e6)
    return b

A,B=mk(1)
Bt=B.T
C=torch.empty((1,256),device=DEV,dtype=torch.float16)
W=torch.empty((16,1,256),device=DEV,dtype=torch.float32)
print("torch matmul(A,B.T)   %.2f"%wall(lambda a,b: torch.matmul(a,b.T),(A,B)),flush=True)
print("torch mm out= (preall)%.2f"%wall(lambda: torch.mm(A,Bt,out=C),()),flush=True)
# isolate: just the splitk kernel, preallocated W, no reduce
NB=256//16
def only_gemm():
    gsk[(NB*16,)](A,B,W,1,A.stride(0),B.stride(0),BLOCK_M=16,BLOCK_N=16,BLOCK_K=64,
                  SPLIT_K=16,NB=NB,K=7168,NN=256,num_warps=2,num_stages=2)
print("gsk only (prealloc)   %.2f"%wall(only_gemm,()),flush=True)
def only_red():
    gred[(1,)](W,C,256,SPLIT_K=16,BLOCK=1024,num_warps=4)
print("gred only (prealloc)  %.2f"%wall(only_red,()),flush=True)
def both():
    only_gemm(); only_red()
print("both  (prealloc)      %.2f"%wall(both,()),flush=True)
def both_alloc():
    M=A.shape[0]
    c=torch.empty((M,256),device=DEV,dtype=torch.float16)
    w=torch.empty((16,M,256),device=DEV,dtype=torch.float32)
    gsk[(NB*16,)](A,B,w,M,A.stride(0),B.stride(0),BLOCK_M=16,BLOCK_N=16,BLOCK_K=64,
                  SPLIT_K=16,NB=NB,K=7168,NN=256,num_warps=2,num_stages=2)
    gred[(1,)](w,c,M*256,SPLIT_K=16,BLOCK=1024,num_warps=4)
    return c
print("both + allocs         %.2f"%wall(both_alloc,()),flush=True)
