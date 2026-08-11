import json,glob,os
from collections import Counter,defaultdict
ROOT='/home/qinwu/dev/solbench-dev/main'
cand={}
for f in sorted(glob.glob(ROOT+'/artifacts/06/candidates/*.json')):
    d=json.load(open(f)); cand[d.get('problem') or os.path.basename(f)[:-5]]=d
VAR=['v1_eager','v2_compile','v3_compile_max_autotune','v4_contiguous','v5_compile_contiguous']
inv={}
for v in VAR:
    s=set()
    for p,d in cand.items():
        for fl in ((d.get('variants') or {}).get(v) or {}).get('failures') or []:
            if fl['status']=='INVALID_REFERENCE': s.add((p,fl['workload_uuid']))
    inv[v]=s
base=inv['v1_eager']
for v in VAR: print(v,len(inv[v]),'identical to v1_eager set:',inv[v]==base)
probs=defaultdict(int)
for p,u in base: probs[p]+=1
print('\nINVALID_REFERENCE problems:',len(probs))
for p in sorted(probs): print(f'  {p}: {probs[p]}')
# logs?
logs=set()
for p,d in cand.items():
    for fl in ((d.get('variants') or {}).get('v1_eager') or {}).get('failures') or []:
        if fl['status']=='INVALID_REFERENCE': logs.add(fl.get('log') or '')
print('distinct log strings:',{l[:200] for l in logs})
# manifest deferred?
man=json.load(open(ROOT+'/artifacts/09/manifest-v1.2.json'))
for p in sorted(probs):
    mp=man['problems'].get(p)
    if mp is None: print(p,'NOT IN MANIFEST'); continue
    sc=Counter(bool(w.get('scoreable')) for w in mp.get('workloads',{}).values())
    print(p,'manifest workloads',len(mp.get('workloads',{})),'scoreable',sc[True],'reason',str(mp.get('deferred_reason') or mp.get('exclusion_reason') or mp.get('note'))[:120])
