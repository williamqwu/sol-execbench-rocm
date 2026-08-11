import json,glob,os
from collections import Counter,defaultdict
ROOT='/home/qinwu/dev/solbench-dev/main'
st=json.load(open('/home/qinwu/dev/solbench-dev/main/artifacts/11/compile-diag/taxonomy/state.json'))
EPS=1.1920928955078125e-07
# load all workload metadata
wl={}
for f in glob.glob(ROOT+'/artifacts/05/workloads/*/*/workload.jsonl'):
    parts=f.split('/'); cat=parts[-3]; name=parts[-2]; pkey=f"{cat}__{name}"
    for line in open(f):
        d=json.loads(line); wl[(pkey,d['uuid'])]=d
print('workload metadata rows:',len(wl))
for vn in ['v2','v3']:
    S=st[vn]
    tab=Counter()
    for pk,d in S.items():
        for u,s in d.items():
            m=wl.get((pk,u))
            if m is None: tab[('NOMETA',s)]+=1; continue
            t=m['tolerance']
            floored = (t['max_rtol']<=EPS*1.0000001) and (t['max_atol']<=EPS*1.0000001)
            rtol_floored = t['max_rtol']<=EPS*1.0000001
            tab[(('rtol_at_fp32_eps' if rtol_floored else 'rtol_looser'),s)]+=1
    print('\n',vn, dict(tab))
# per-problem: does rtol-floored separate pass/fail perfectly?
print('\n### separation test on split problems (v2)')
print('| problem | n | fail | fail&rtol_floored | pass&rtol_floored | perfect separation by rtol==fp32eps? |')
print('|---'*6+'|')
nperf=0; ntot=0
for pk,d in sorted(st['v2'].items()):
    fails={u for u,s in d.items() if s not in ('PASSED','NO_RECORD')}
    passes={u for u,s in d.items() if s=='PASSED'}
    if not fails or not passes: continue
    ntot+=1
    fl={u for u in d if (pk,u) in wl and wl[(pk,u)]['tolerance']['max_rtol']<=EPS*1.0000001}
    perfect = (fails<=fl) and (passes&fl==set())
    nperf+=perfect
    print(f"| {pk} | {len(d)} | {len(fails)} | {len(fails&fl)} | {len(passes&fl)} | {perfect} |")
print('perfect:',nperf,'of',ntot)
