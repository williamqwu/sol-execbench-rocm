import json,glob,os,sqlite3
from collections import Counter,defaultdict
ROOT='/home/qinwu/dev/solbench-dev/main'
con=sqlite3.connect(ROOT+'/leaderboard/solbench.db')
db=defaultdict(dict)
for sid,pk,u,st in con.execute("select submission_id,problem_key,workload_uuid,status from result where submission_id<=4"):
    db[sid][(pk,u)]=st
V={1:'v1_eager',2:'v2_compile',3:'v3_compile_max_autotune',4:'v4_contiguous'}
def load(src):
    out={}
    for f in sorted(glob.glob(src+'/*.json')):
        doc=json.load(open(f)); pkey=doc.get('problem') or os.path.basename(f)[:-5]
        out[pkey]=doc
    return out
cand=load(ROOT+'/artifacts/06/candidates'); auth=load(ROOT+'/artifacts/06/authoritative')
for sid,vn in V.items():
    cpass={(p,u) for p,d in cand.items() for u in (((d.get('variants') or {}).get(vn) or {}).get('latency_ms_by_workload') or {})}
    apass={(p,u) for p,d in auth.items() for u in (((d.get('variants') or {}).get(vn) or {}).get('latency_ms_by_workload') or {})}
    rows=set(db[sid]); dbf={k for k,v in db[sid].items() if v=='FAILED'}
    print(vn,'dbrows',len(rows),'candpass',len(cpass),'authpass',len(apass))
    print('   rows==candpass?',rows==cpass, ' rows-candpass',len(rows-cpass),' candpass-rows',len(cpass-rows))
    print('   dbFAILED',len(dbf),' == candpass-authpass?',dbf==(cpass-apass), len(cpass-apass))
    # old all_passed theory
    badprob={p for p,d in cand.items() if ((d.get('variants') or {}).get(vn) or {}).get('all_passed') is False}
    theory={k for k in cpass if k[0] in badprob}
    print('   all_passed-theory FAILED:',len(theory),'match?',theory==dbf)
