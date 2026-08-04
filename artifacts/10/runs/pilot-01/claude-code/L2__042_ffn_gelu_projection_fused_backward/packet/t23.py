from lt import *
import torch, numpy as np, itertools
torch.manual_seed(0)
H=512
x=torch.randn(256,H,device=dev)
ref=x.sum(-1).cpu().numpy()
X=x.cpu().numpy()
def tree(a):
    a=list(a)
    while len(a)>1:
        a=[np.float32(a[i]+a[i+1]) if i+1<len(a) else a[i] for i in range(0,len(a),2)]
    return a[0]
def seq(a):
    s=np.float32(0)
    for v in a: s=np.float32(s+v)
    return s
best=None
# model: nt threads; thread t handles indices in "vec4 chunks strided by nt"
for nt in [32,64,128,256,512]:
  for vec in [1,2,4]:
    for inner in ['seq','tree']:
      f = seq if inner=='seq' else tree
      parts=[]
      ok=True
      nchunk=H//vec
      for t in range(nt):
          idxs=[]
          for cb in range(t, nchunk, nt):
              idxs.extend(range(cb*vec,(cb+1)*vec))
          if not idxs: continue
          parts.append(f([X[0][i] for i in idxs]))
      for outer in ['seq','tree']:
          g=(seq if outer=='seq' else tree)(parts)
          if g==ref[0]:
              print("MATCH row0:",nt,vec,inner,outer)
              # verify all rows
              allok=True
              for r in range(256):
                  ps=[]
                  for t in range(nt):
                      idxs=[]
                      for cb in range(t,nchunk,nt):
                          idxs.extend(range(cb*vec,(cb+1)*vec))
                      if idxs: ps.append(f([X[r][i] for i in idxs]))
                  if (seq if outer=='seq' else tree)(ps)!=ref[r]: allok=False; break
              print("   all rows:",allok)
