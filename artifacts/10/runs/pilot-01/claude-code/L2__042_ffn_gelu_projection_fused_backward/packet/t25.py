from lt import *
import torch, numpy as np
torch.manual_seed(0)
H=512
x=torch.randn(1024,H,device=dev)
ref=x.sum(-1).cpu().numpy()
X=x.cpu().numpy().astype(np.float32)
f=np.float32
# torch ReduceOp model: nt threads along reduction, vec=4 contiguous per thread,
# per-thread seq combine of 4 accumulators, then shared-mem halving down to warpSize=64,
# then balanced tree over 64.
def model(row, nt=128, vec=4, warp=64):
    n=len(row); nchunk=n//vec
    s=[]
    for t in range(nt):
        acc=[f(0.0)]*vec
        idx=t
        while idx < nchunk:
            for i in range(vec): acc[i]=f(acc[i]+row[idx*vec+i])
            idx+=nt
        v=acc[0]
        for i in range(1,vec): v=f(v+acc[i])
        s.append(v)
    dim=nt
    while dim>warp:
        off=dim//2
        s=[f(s[i]+s[i+off]) if i+off<dim else s[i] for i in range(off)]+s[off:]
        dim=off
    v=s[:warp]
    off=1
    while off<warp:
        v=[f(v[i]+v[i+off]) if i+off<warp else v[i] for i in range(warp)]
        off*=2
    return v[0]
for nt in [64,128,256]:
    got=np.array([model(X[i],nt=nt) for i in range(200)])
    print(f"nt={nt}: match {(got==ref[:200]).mean():.4f}")
