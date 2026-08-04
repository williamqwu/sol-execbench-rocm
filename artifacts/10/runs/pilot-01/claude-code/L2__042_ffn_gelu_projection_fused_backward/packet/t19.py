from lt import *
import torch, numpy as np
torch.manual_seed(0)
H=512
x=torch.randn(64,H,device=dev)
ref=x.sum(-1).cpu().numpy()
x32=x.cpu().numpy()
def tree(a):
    a=list(a)
    while len(a)>1:
        a=[np.float32(a[i]+a[i+1]) if i+1<len(a) else a[i] for i in range(0,len(a),2)]
    return a[0]
got=np.array([tree(x32[i]) for i in range(64)])
print("tree match frac", (got==ref).mean())
# maybe torch uses 4-way unroll per thread then tree
def unroll_then_tree(a, nt, ur):
    # nt threads, each does ur sequential elements contiguously chunked? 
    parts=[]
    for t in range(nt):
        acc=[np.float32(0)]*ur
        i=t*ur
        # vectorized load of ur contiguous
        for j in range(ur):
            if i+j<len(a): acc[j]=np.float32(acc[j]+a[i+j])
        parts.append(tree(acc))
    return tree(parts)
for nt,ur in [(128,4),(64,8),(256,2)]:
    g=np.array([unroll_then_tree(x32[i],nt,ur) for i in range(64)])
    print(f"nt{nt} ur{ur}", (g==ref).mean())
