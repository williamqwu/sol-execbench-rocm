import json,glob,os
from collections import Counter,defaultdict
ROOT='/home/qinwu/dev/solbench-dev/main'
man=json.load(open(ROOT+'/artifacts/09/manifest-v1.2.json'))
b={}
for key,p in man['problems'].items():
    for uuid,w in p.get('workloads',{}).items():
        if w.get('scoreable') and w.get('t_sol_ms') and w.get('t_b_ms'):
            b[(key,uuid)]=(w['t_sol_ms'],w['t_b_ms'])
print('bounds (scoreable w/ t_sol & t_b):',len(b),'problems:',len({k[0] for k in b}))
# scoreable count regardless of t_sol/t_b
sc=sum(1 for key,p in man['problems'].items() for u,w in p.get('workloads',{}).items() if w.get('scoreable'))
print('scoreable total:',sc, 'all workloads:',sum(len(p.get('workloads',{})) for p in man['problems'].values()))

per_variant=defaultdict(dict)
for src,label in ((ROOT+'/artifacts/06/candidates','sweep'),(ROOT+'/artifacts/06/authoritative','auth')):
    for f in sorted(glob.glob(src+'/*.json')):
        doc=json.load(open(f)); pkey=doc.get('problem') or os.path.basename(f)[:-5]
        for vname,v in (doc.get('variants') or {}).items():
            store=per_variant[vname]
            for fl in v.get('failures') or []:
                u=fl.get('workload_uuid')
                if u: store[(pkey,u)]=(None,label,fl.get('status') or 'FAILED')
            for u,ms in (v.get('latency_ms_by_workload') or {}).items():
                store[(pkey,u)]=(ms,label,'PASSED')
            if v.get('error') and not v.get('failures') and not v.get('latency_ms_by_workload'):
                pass
for vname in sorted(per_variant):
    e=per_variant[vname]
    inb={k:v for k,v in e.items() if k in b}
    c=Counter(v[2] for v in inb.values())
    print(f"{vname}: entries={len(e)} in-bounds={len(inb)} probs={len({k[0] for k in inb})} {dict(c)}")
