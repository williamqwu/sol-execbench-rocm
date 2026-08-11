import json,glob,statistics
ROOT='/home/qinwu/dev/solbench-dev/main'
cand={}
for f in sorted(glob.glob(ROOT+'/artifacts/06/candidates/*.json')):
    d=json.load(open(f)); cand[d['problem']]=d
order={}
for f in glob.glob(ROOT+'/artifacts/05/workloads/*/*/workload.jsonl'):
    p=f.split('/'); order[f"{p[-3]}__{p[-2]}"]=[json.loads(l)['uuid'] for l in open(f)]
# Only problems whose v2 failure set is exactly the first 8 (54 problems)
lo,hi=[],[]
sel=[]
for pk,d in cand.items():
    v1=(d['variants'].get('v1_eager') or {}).get('latency_ms_by_workload') or {}
    v2v=d['variants'].get('v2_compile') or {}
    v2=v2v.get('latency_ms_by_workload') or {}
    fails={f['workload_uuid'] for f in v2v.get('failures') or []}
    o=[u for u in order.get(pk,[]) if u in v1]
    if not o: continue
    idx={u:i for i,u in enumerate(o)}
    fi=sorted(idx[u] for u in fails if u in idx)
    if fi!=list(range(8)): continue
    sel.append(pk)
    for u in o:
        if u in v2 and u in v1 and v1[u]>0:
            r=v2[u]/v1[u]
            (lo if idx[u]<8 else hi).append(r)
def s(x): return f"n={len(x)} median={statistics.median(x):.4f} mean={statistics.mean(x):.4f} p10={sorted(x)[len(x)//10]:.4f} p90={sorted(x)[9*len(x)//10]:.4f}"
print('problems selected (v2 fails exactly idx0-7):',len(sel))
print('index 0-7  (COMPILED, all FAILED -> no v2 latency recorded):', s(lo) if lo else 'none')
print('index >=8  (PASSED)  v2/v1 latency ratio:', s(hi))
frac=sum(1 for r in hi if 0.97<=r<=1.03)/len(hi)
print(f'fraction of index>=8 ratios within +-3% of eager: {frac:.3f}  ({sum(1 for r in hi if 0.97<=r<=1.03)}/{len(hi)})')
# contrast: problems where v2 passed everything (never hit limit? no -- all workloads compiled)
allp=[];
for pk,d in cand.items():
    v2v=d['variants'].get('v2_compile') or {}
    if v2v.get('failures'): continue
    v1=(d['variants'].get('v1_eager') or {}).get('latency_ms_by_workload') or {}
    v2=v2v.get('latency_ms_by_workload') or {}
    o=[u for u in order.get(pk,[]) if u in v1]
    idx={u:i for i,u in enumerate(o)}
    for u in o:
        if u in v2 and v1.get(u):
            allp.append((idx[u], v2[u]/v1[u]))
a=[r for i,r in allp if i<8]; b=[r for i,r in allp if i>=8]
print('\nclean-pass problems, idx<8 :',s(a))
print('clean-pass problems, idx>=8:',s(b))
print('  within +-3% idx<8:', f"{sum(1 for r in a if 0.97<=r<=1.03)/len(a):.3f}", ' idx>=8:', f"{sum(1 for r in b if 0.97<=r<=1.03)/len(b):.3f}")
