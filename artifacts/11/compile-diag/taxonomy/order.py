import json,glob
from collections import Counter
ROOT='/home/qinwu/dev/solbench-dev/main'
st=json.load(open('/home/qinwu/dev/solbench-dev/main/artifacts/11/compile-diag/taxonomy/state.json'))
order={}
for f in glob.glob(ROOT+'/artifacts/05/workloads/*/*/workload.jsonl'):
    p=f.split('/'); pkey=f"{p[-3]}__{p[-2]}"
    order[pkey]=[json.loads(l)['uuid'] for l in open(f)]
for vn in ['v2','v3']:
    print('=====',vn)
    c=Counter(); rows=[]
    for pk,d in sorted(st[vn].items()):
        o=[u for u in order[pk] if u in d]
        idx={u:i for i,u in enumerate(o)}
        fails=sorted(idx[u] for u,s in d.items() if s not in ('PASSED','NO_RECORD'))
        norec=sorted(idx[u] for u,s in d.items() if s=='NO_RECORD')
        if not fails and not norec: continue
        prefix = fails==list(range(len(fails)))
        c['fail_is_leading_prefix' if prefix else 'other']+=1
        rows.append((pk,len(o),fails,norec,prefix))
    print(dict(c))
    print('| problem | n | failing indices (0-based, jsonl order) | no-record indices | leading prefix? |')
    print('|---'*5+'|')
    for pk,n,f,nr,pre in rows:
        fs=','.join(map(str,f)) if len(f)<=20 else f"{f[0]}..{f[-1]} ({len(f)})"
        print(f"| {pk} | {n} | {fs} | {','.join(map(str,nr))} | {pre} |")
    print()
