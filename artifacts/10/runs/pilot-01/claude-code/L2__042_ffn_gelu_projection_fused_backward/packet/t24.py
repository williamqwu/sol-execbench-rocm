from lt import *
import torch, numpy as np
torch.manual_seed(0)
H=512
x=torch.randn(512,H,device=dev)
ref=x.sum(-1).cpu().numpy()
X=x.cpu().numpy().astype(np.float32)

def f32(a): return np.float32(a)
# model: 512 threads, 1 val each. warp(64) shuffle-down tree offsets 32,16,8,4,2,1
# then 8 warp partials -> reduced by warp 0 similarly
def model(row, nthread=512, warp=64):
    vals=list(row[:nthread])
    nw=nthread//warp
    warps=[]
    for w in range(nw):
        v=vals[w*warp:(w+1)*warp]
        off=warp//2
        while off>=1:
            v=[f32(v[i]+v[i+off]) for i in range(off)]+v[off:]
            off//=2
        warps.append(v[0])
    # cross-warp tree
    v=warps[:]
    off=len(v)//2
    while off>=1:
        v=[f32(v[i]+v[i+off]) for i in range(off)]+v[off:]
        off//=2
    return v[0]
got=np.array([model(X[i]) for i in range(512)])
print("warp-shuffle model match frac:", (got==ref).mean())
# variant: cross-warp seq
def model2(row):
    warp=64; nw=8; warps=[]
    for w in range(nw):
        v=list(row[w*warp:(w+1)*warp]); off=32
        while off>=1:
            v=[f32(v[i]+v[i+off]) for i in range(off)]+v[off:]; off//=2
        warps.append(v[0])
    s=np.float32(0)
    for a in warps: s=f32(s+a)
    return s
got2=np.array([model2(X[i]) for i in range(512)])
print("warp+seq model:", (got2==ref).mean())
