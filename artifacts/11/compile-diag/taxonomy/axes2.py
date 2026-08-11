import json,glob
from collections import Counter,defaultdict
ROOT='/home/qinwu/dev/solbench-dev/main'
st=json.load(open('/home/qinwu/dev/solbench-dev/main/artifacts/11/compile-diag/taxonomy/state.json'))
wl={}
for f in glob.glob(ROOT+'/artifacts/05/workloads/*/*/workload.jsonl'):
    p=f.split('/'); pkey=f"{p[-3]}__{p[-2]}"
    for line in open(f):
        d=json.loads(line); wl[(pkey,d['uuid'])]=d
# per split problem, find axis whose value set separates
res=Counter(); detail={}
for pk,d in sorted(st['v2'].items()):
    fails={u for u,s in d.items() if s not in ('PASSED','NO_RECORD')}
    passes={u for u,s in d.items() if s=='PASSED'}
    if not fails or not passes: continue
    axnames=set()
    for u in d:
        m=wl.get((pk,u))
        if m: axnames|=set(m['axes'])
    found=[]
    for ax in sorted(axnames):
        fv={wl[(pk,u)]['axes'].get(ax) for u in fails if (pk,u) in wl}
        pv={wl[(pk,u)]['axes'].get(ax) for u in passes if (pk,u) in wl}
        if fv and pv and not (fv & pv):
            found.append((ax,sorted(map(str,fv)),sorted(map(str,pv))))
    # also: is order (jsonl line index) a separator -> first half / second half?
    detail[pk]=(sorted(axnames),found)
    res['separated' if found else 'not_separated']+=1
print(res)
print()
print('| problem | axes | separating axis | FAIL values | PASS values |')
print('|---'*5+'|')
for pk,(ax,found) in detail.items():
    if found:
        for a,fv,pv in found:
            print(f"| {pk} | {','.join(ax)} | **{a}** | {','.join(fv)} | {','.join(pv)} |")
print()
print('--- NOT separated by any single axis ---')
for pk,(ax,found) in detail.items():
    if not found: print(' ',pk,ax)
