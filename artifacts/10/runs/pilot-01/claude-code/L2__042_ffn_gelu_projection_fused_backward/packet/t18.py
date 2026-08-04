from lt import *
import torch, numpy as np
torch.manual_seed(0)
H=512
x=torch.randn(4,H,device=dev)
ref=x.sum(-1)
xc=x.cpu().numpy().astype(np.float64)
x32=x.cpu().numpy()
# candidate orders
def seq(a):
    s=np.float32(0)
    for v in a: s=np.float32(s+v)
    return s
def tree(a):
    a=list(a)
    while len(a)>1:
        a=[np.float32(a[i]+a[i+1]) if i+1<len(a) else a[i] for i in range(0,len(a),2)]
    return a[0]
def striped(a, nt):
    # nt threads each accumulate strided, then tree
    parts=[seq(a[t::nt]) for t in range(nt)]
    return tree(parts)
for nm,f in [('seq',seq),('tree',tree)]:
    got=np.array([f(x32[i]) for i in range(4)])
    print(nm, np.array_equal(got, ref.cpu().numpy()), got[0], ref[0].item())
for nt in [32,64,128,256,512]:
    got=np.array([striped(x32[i],nt) for i in range(4)])
    print(f"striped{nt}", np.array_equal(got, ref.cpu().numpy()))
