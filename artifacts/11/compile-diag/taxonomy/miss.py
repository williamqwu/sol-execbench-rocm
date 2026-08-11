import json,glob,os
from collections import defaultdict
ROOT='/home/qinwu/dev/solbench-dev/main'
man=json.load(open(ROOT+'/artifacts/09/manifest-v1.2.json'))
by_prob={k:{u for u,w in p.get('workloads',{}).items() if w.get('scoreable')} for k,p in man['problems'].items()}
by_prob={k:v for k,v in by_prob.items() if v}
def load(src):
    out={}
    for f in sorted(glob.glob(src+'/*.json')):
        d=json.load(open(f)); out[d.get('problem') or os.path.basename(f)[:-5]]=d
    return out
cand=load(ROOT+'/artifacts/06/candidates'); gap=load(ROOT+'/artifacts/06/candidates-gapfill'); auth=load(ROOT+'/artifacts/06/authoritative')
print('gapfill problems:',sorted(gap))
for v in ['v2_compile','v3_compile_max_autotune']:
    print('\n=====',v)
    for pk in sorted(by_prob):
        vv=(cand[pk].get('variants') or {}).get(v) or {}
        seen=set(vv.get('latency_ms_by_workload') or {})|{f['workload_uuid'] for f in vv.get('failures') or []}
        missing=by_prob[pk]-seen
        if not missing: continue
        g=(gap.get(pk,{}).get('variants') or {}).get(v) or {}
        gseen=set(g.get('latency_ms_by_workload') or {})|{f['workload_uuid'] for f in g.get('failures') or []}
        a=(auth.get(pk,{}).get('variants') or {}).get(v) or {}
        aseen=set(a.get('latency_ms_by_workload') or {})|{f['workload_uuid'] for f in a.get('failures') or []}
        print(f"  {pk}: scoreable={len(by_prob[pk])} cand_seen={len(seen)} MISSING={len(missing)}"
              f" | cand.error={str(vv.get('error'))[:90]!r} | gapfill_covers={len(missing&gseen)} auth_covers={len(missing&aseen)}")
