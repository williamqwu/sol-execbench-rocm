import json,glob,os,sys
from collections import Counter, defaultdict
ROOT='/home/qinwu/dev/solbench-dev/main'
DIRS={'candidates':'artifacts/06/candidates','gapfill':'artifacts/06/candidates-gapfill','authoritative':'artifacts/06/authoritative'}
VARIANTS=['v1_eager','v2_compile','v3_compile_max_autotune','v4_contiguous','v5_compile_contiguous']
def cat(pkey): return pkey.split('__')[0]
data={}
for tag,d in DIRS.items():
    for f in sorted(glob.glob(os.path.join(ROOT,d,'*.json'))):
        doc=json.load(open(f))
        pkey=doc.get('problem') or os.path.basename(f)[:-5]
        data.setdefault(tag,{})[pkey]=doc

print("=== FILE COUNTS ===")
for tag in DIRS: print(tag, len(data.get(tag,{})))

print("\n=== per-dir, per-variant failure status counts ===")
for tag in DIRS:
    print('---',tag)
    for v in VARIANTS:
        c=Counter(); tot_wl=0; tot_pass=0; nprob=0; nerr=0
        for pkey,doc in data[tag].items():
            vv=(doc.get('variants') or {}).get(v)
            if vv is None: continue
            nprob+=1
            tot_wl+=vv.get('workloads') or 0
            tot_pass+=vv.get('passed') or 0
            if vv.get('error'): nerr+=1
            for fl in vv.get('failures') or []:
                c[fl.get('status') or 'FAILED']+=1
        print(f"  {v}: problems={nprob} workloads={tot_wl} passed={tot_pass} errors={nerr} failures={sum(c.values())} {dict(c)}")
