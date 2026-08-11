import json,glob
from collections import Counter
ROOT='/home/qinwu/dev/solbench-dev/main'
EPS=1.1920928955078125e-07
wl={}; order={}
for f in glob.glob(ROOT+'/artifacts/05/workloads/*/*/workload.jsonl'):
    p=f.split('/'); pk=f"{p[-3]}__{p[-2]}"; order[pk]=[]
    for line in open(f):
        d=json.loads(line); wl[(pk,d['uuid'])]=d; order[pk].append(d['uuid'])
st=json.load(open('/home/qinwu/dev/solbench-dev/main/artifacts/11/compile-diag/taxonomy/state.json'))
t=Counter()
for pk,d in st['v2'].items():
    idx={u:i for i,u in enumerate([u for u in order[pk] if u in d])}
    for u,s in d.items():
        if idx[u]>=8: continue
        tight = wl[(pk,u)]['tolerance']['max_rtol']<=EPS*1.0000001
        t[('tight' if tight else 'loose', 'FAIL' if s not in ('PASSED','NO_RECORD') else s)]+=1
print('v2, idx<8 only:', dict(t))
for g in ['tight','loose']:
    n=sum(v for k,v in t.items() if k[0]==g); f=t[(g,'FAIL')]
    print(f"  {g}: {n} workloads, {f} failed ({100*f/n:.1f}%)")
# global fallback share
tot=sum(len(d) for d in st['v2'].values()); ge=sum(1 for pk,d in st['v2'].items() for u in d if [x for x in order[pk] if x in d].index(u)>=8)
print(f"\nscoreable workloads at jsonl idx>=8 (never compiled): {ge} of {tot} = {100*ge/tot:.1f}%")
# v3 RUNTIME_ERROR detail
cand={}
for f in sorted(glob.glob(ROOT+'/artifacts/06/candidates/*.json')):
    d=json.load(open(f)); cand[d['problem']]=d
print('\nv3 RUNTIME_ERROR occurrences:')
for pk,d in sorted(cand.items()):
    fl=[f for f in (d['variants']['v3_compile_max_autotune'].get('failures') or []) if f['status']=='RUNTIME_ERROR']
    if fl: print(' ',pk,len(fl),repr((fl[0].get('log') or '')[:300]))
print('\nv5 RUNTIME_ERROR sample log:')
for pk,d in sorted(cand.items()):
    fl=[f for f in (d['variants']['v5_compile_contiguous'].get('failures') or []) if f['status']=='RUNTIME_ERROR']
    if fl: print(' ',pk,repr((fl[0].get('log') or '')[:400])); break
