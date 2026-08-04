from lt import *
import reference, kernel, json, torch
wls=[json.loads(l) for l in open('workload.jsonl')]
allok=True
for w in wls:
    B=w['axes']['batch_size']; S=w['axes']['seq_len']
    atol=w['tolerance']['max_atol']; rtol=w['tolerance']['max_rtol']
    inp=gen(B,S,seed=B*1000+S)
    ref=reference.run(**inp)
    out=kernel.run(**inp)
    worst=1.0
    for n,a,b in zip(names,out,ref):
        a=a.float(); b=b.float()
        if a.shape!=b.shape: print("SHAPE MISMATCH",n,a.shape,b.shape); worst=0; continue
        e=(a-b).abs(); tolm=atol+rtol*b.abs()
        m=(e<=tolm).float().mean().item()
        if m<worst: worst=m; wn=n; wm=e.max().item()
    st='PASS' if worst>=0.99 else 'FAIL'
    if worst<0.99: allok=False
    print(f"B={B:3d} S={S:5d} {st} worst={worst:.5f} ({wn if worst<1 else '-'})")
print("ALL OK" if allok else "SOME FAIL")
