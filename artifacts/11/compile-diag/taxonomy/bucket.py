import json,glob
from collections import Counter
ROOT='/home/qinwu/dev/solbench-dev/main'
st=json.load(open('/home/qinwu/dev/solbench-dev/main/artifacts/11/compile-diag/taxonomy/state.json'))
order={}
for f in glob.glob(ROOT+'/artifacts/05/workloads/*/*/workload.jsonl'):
    p=f.split('/'); order[f"{p[-3]}__{p[-2]}"]=[json.loads(l)['uuid'] for l in open(f)]
for vn in ['v2','v3']:
    t=Counter()
    for pk,d in st[vn].items():
        o=[u for u in order[pk] if u in d]; idx={u:i for i,u in enumerate(o)}
        for u,s in d.items():
            t[('idx<8' if idx[u]<8 else 'idx>=8', s)]+=1
    print(vn, dict(t))
    lt=sum(v for k,v in t.items() if k[0]=='idx<8'); ge=sum(v for k,v in t.items() if k[0]=='idx>=8')
    fl=sum(v for k,v in t.items() if k[0]=='idx<8' and k[1] not in('PASSED','NO_RECORD'))
    fg=sum(v for k,v in t.items() if k[0]=='idx>=8' and k[1] not in('PASSED','NO_RECORD'))
    print(f"   idx<8: {lt} workloads, {fl} failed ({100*fl/lt:.1f}%)  | idx>=8: {ge} workloads, {fg} failed ({100*fg/ge:.1f}%)")
