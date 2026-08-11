import json,glob,os,sqlite3
from collections import Counter,defaultdict
ROOT='/home/qinwu/dev/solbench-dev/main'
man=json.load(open(ROOT+'/artifacts/09/manifest-v1.2.json'))
by_prob={k:{u for u,w in p.get('workloads',{}).items() if w.get('scoreable')} for k,p in man['problems'].items()}
by_prob={k:v for k,v in by_prob.items() if v}
cand={}
for f in sorted(glob.glob(ROOT+'/artifacts/06/candidates/*.json')):
    d=json.load(open(f)); cand[d['problem']]=d
def state(v):
    out={}
    for pk,S in by_prob.items():
        vv=(cand[pk].get('variants') or {}).get(v) or {}
        lm=set(vv.get('latency_ms_by_workload') or {})
        fl={f['workload_uuid']:f['status'] for f in vv.get('failures') or []}
        d={}
        for u in S:
            d[u]='PASSED' if u in lm else fl.get(u,'NO_RECORD')
        out[pk]=d
    return out
s2=state('v2_compile'); s3=state('v3_compile_max_autotune')
def cls(d):
    n=len(d); p=sum(1 for x in d.values() if x=='PASSED')
    if p==n: return 'all_pass'
    if p==0: return 'all_fail'
    return 'split'
c2=Counter(cls(d) for d in s2.values()); c3=Counter(cls(d) for d in s3.values())
print('v2 problem classes:',dict(c2)); print('v3 problem classes:',dict(c3))
print()
print("| problem | cat | n | v2 pass | v2 fail | v2 norec | v2 class | v3 pass | v3 fail | v3 norec | v3 class |")
print('|---'*11+'|')
probs=sorted(pk for pk in by_prob if cls(s2[pk])!='all_pass' or cls(s3[pk])!='all_pass')
for pk in probs:
    a,b=s2[pk],s3[pk]
    f=lambda d,s: sum(1 for x in d.values() if x==s)
    fail=lambda d: sum(1 for x in d.values() if x not in ('PASSED','NO_RECORD'))
    print(f"| {pk} | {pk.split('__')[0]} | {len(a)} | {f(a,'PASSED')} | {fail(a)} | {f(a,'NO_RECORD')} | {cls(a)} | {f(b,'PASSED')} | {fail(b)} | {f(b,'NO_RECORD')} | {cls(b)} |")
print()
print('n problems with any v2 non-pass:',len([p for p in by_prob if cls(s2[p])!='all_pass']))
print('n problems with any v3 non-pass:',len([p for p in by_prob if cls(s3[p])!='all_pass']))
json.dump({'v2':s2,'v3':s3},open('/home/qinwu/dev/solbench-dev/main/artifacts/11/compile-diag/taxonomy/state.json','w'))
