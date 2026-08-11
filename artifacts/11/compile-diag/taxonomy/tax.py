import json,glob,os
from collections import Counter,defaultdict
ROOT='/home/qinwu/dev/solbench-dev/main'
man=json.load(open(ROOT+'/artifacts/09/manifest-v1.2.json'))
scoreable={(k,u) for k,p in man['problems'].items() for u,w in p.get('workloads',{}).items() if w.get('scoreable')}
by_prob=defaultdict(set)
for k,u in scoreable: by_prob[k].add(u)
def load(src):
    out={}
    for f in sorted(glob.glob(src+'/*.json')):
        d=json.load(open(f)); out[d.get('problem') or os.path.basename(f)[:-5]]=d
    return out
cand=load(ROOT+'/artifacts/06/candidates')
gap=load(ROOT+'/artifacts/06/candidates-gapfill')
VAR=['v1_eager','v2_compile','v3_compile_max_autotune','v4_contiguous','v5_compile_contiguous']
def cat(p): return p.split('__')[0]
CATS=['L1','L2','Quant','FlashInfer-Bench']

print("### Table 1: candidates/ sweep, status x variant x category (ALL workloads incl. non-scoreable)")
hdr="| variant | category | workloads | PASSED | INCORRECT_NUMERICAL | INVALID_REFERENCE | RUNTIME_ERROR | no record |"
print(hdr); print('|---'*8+'|')
for v in VAR:
    for c in CATS:
        n=p=inc=invr=rte=0; nore=0
        for pk,d in cand.items():
            if cat(pk)!=c: continue
            vv=(d.get('variants') or {}).get(v) or {}
            tot=len(man['problems'].get(pk,{}).get('workloads',{}))
            seen=len(vv.get('latency_ms_by_workload') or {})+len(vv.get('failures') or [])
            n+=tot; nore+=tot-seen
            p+=len(vv.get('latency_ms_by_workload') or {})
            for fl in vv.get('failures') or []:
                s=fl['status']
                if s=='INCORRECT_NUMERICAL': inc+=1
                elif s=='INVALID_REFERENCE': invr+=1
                elif s=='RUNTIME_ERROR': rte+=1
        print(f"| {v} | {c} | {n} | {p} | {inc} | {invr} | {rte} | {nore} |")

print()
print("### Table 2: SCOREABLE-only (the 3717), candidates sweep")
print("| variant | category | scoreable | PASSED | INCORRECT_NUMERICAL | RUNTIME_ERROR | no record |")
print('|---'*7+'|')
for v in VAR:
    tot_row=[0]*5
    for c in CATS:
        n=p=inc=rte=nore=0
        for pk in by_prob:
            if cat(pk)!=c: continue
            vv=(cand[pk].get('variants') or {}).get(v) or {}
            lm=vv.get('latency_ms_by_workload') or {}
            fails={f['workload_uuid']:f['status'] for f in vv.get('failures') or []}
            for u in by_prob[pk]:
                n+=1
                if u in lm: p+=1
                elif u in fails:
                    s=fails[u]
                    if s=='INCORRECT_NUMERICAL': inc+=1
                    elif s=='RUNTIME_ERROR': rte+=1
                else: nore+=1
        print(f"| {v} | {c} | {n} | {p} | {inc} | {rte} | {nore} |")
        for i,x in enumerate([n,p,inc,rte,nore]): tot_row[i]+=x
    print(f"| **{v}** | **ALL** | **{tot_row[0]}** | **{tot_row[1]}** | **{tot_row[2]}** | **{tot_row[3]}** | **{tot_row[4]}** |")
