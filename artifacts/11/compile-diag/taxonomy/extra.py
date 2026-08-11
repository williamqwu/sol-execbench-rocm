import json,glob,statistics
from collections import Counter,defaultdict
ROOT='/home/qinwu/dev/solbench-dev/main'
EPS=1.1920928955078125e-07
wl={}
for f in glob.glob(ROOT+'/artifacts/05/workloads/*/*/workload.jsonl'):
    p=f.split('/'); pkey=f"{p[-3]}__{p[-2]}"
    for line in open(f):
        d=json.loads(line); wl[(pkey,d['uuid'])]=d
st=json.load(open('/home/qinwu/dev/solbench-dev/main/artifacts/11/compile-diag/taxonomy/state.json'))
order={}
for f in glob.glob(ROOT+'/artifacts/05/workloads/*/*/workload.jsonl'):
    p=f.split('/'); order[f"{p[-3]}__{p[-2]}"]=[json.loads(l)['uuid'] for l in open(f)]

print("### tolerance by category, idx<8 only, v2 verdict")
for c in ['L1','L2','Quant','FlashInfer-Bench']:
    rt=[];  fl=0; n=0
    for pk,d in st['v2'].items():
        if not pk.startswith(c+'__'): continue
        idx={u:i for i,u in enumerate([u for u in order[pk] if u in d])}
        for u,s in d.items():
            if idx[u]>=8: continue
            n+=1; rt.append(wl[(pk,u)]['tolerance']['max_rtol'])
            if s not in ('PASSED','NO_RECORD'): fl+=1
    rt.sort()
    at_eps=sum(1 for x in rt if x<=EPS*1.0000001)
    print(f"| {c} | idx<8 n={n} | failed={fl} ({100*fl/n:.1f}%) | median max_rtol={statistics.median(rt):.3e} | at fp32 eps: {at_eps} ({100*at_eps/n:.1f}%) |")

print()
print("### L2__009_decoder_layer_with_residual_connections, jsonl order")
pk='L2__009_decoder_layer_with_residual_connections'
print('| idx | uuid[:8] | axes | max_atol | max_rtol | v1 | v2 | v3 | v4 |')
print('|---'*9+'|')
cand={}
for f in sorted(glob.glob(ROOT+'/artifacts/06/candidates/*.json')):
    d=json.load(open(f)); cand[d['problem']]=d
def verd(pk,v,u):
    vv=cand[pk]['variants'][v]
    if u in (vv.get('latency_ms_by_workload') or {}): return 'pass'
    for f in vv.get('failures') or []:
        if f['workload_uuid']==u: return f['status'][:9].lower()
    return '-'
for i,u in enumerate(order[pk]):
    m=wl[(pk,u)]
    print(f"| {i} | {u[:8]} | {m['axes']} | {m['tolerance']['max_atol']:.3e} | {m['tolerance']['max_rtol']:.3e} | "
          f"{verd(pk,'v1_eager',u)} | {verd(pk,'v2_compile',u)} | {verd(pk,'v3_compile_max_autotune',u)} | {verd(pk,'v4_contiguous',u)} |")

print()
print("### outliers: L2__036, L2__051")
for pk in ['L2__036_convnextv2_layer_with_nhwc_persistence_backward','L2__051_seqlen-finetuned-reconstructed_hyena_complete_forward_block']:
    print(' ',pk)
    for v in ['v1_eager','v2_compile','v3_compile_max_autotune','v4_contiguous']:
        vv=cand[pk]['variants'][v]
        print(f"    {v}: attempted={vv['workloads']} passed={vv['passed']} failstatuses={Counter(f['status'] for f in vv.get('failures') or [])}")
