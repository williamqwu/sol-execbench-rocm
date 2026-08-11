import json,glob,sqlite3
from collections import Counter,defaultdict
ROOT='/home/qinwu/dev/solbench-dev/main'
cand={}
for f in sorted(glob.glob(ROOT+'/artifacts/06/candidates/*.json')):
    d=json.load(open(f)); cand[d['problem']]=d
print("### v1_eager and v4_contiguous INCORRECT_NUMERICAL, per problem")
for v in ['v1_eager','v4_contiguous']:
    print(' --',v)
    for pk,d in sorted(cand.items()):
        fl=[f for f in ((d['variants'].get(v) or {}).get('failures') or []) if f['status']=='INCORRECT_NUMERICAL']
        if fl: print(f'   {pk}: {len(fl)}')
print()
st=json.load(open('/home/qinwu/dev/solbench-dev/main/artifacts/11/compile-diag/taxonomy/state.json'))
s2,s3=st['v2'],st['v3']
def failset(s): return {pk for pk,d in s.items() if any(x not in ('PASSED',) for x in d.values())}
f2,f3=failset(s2),failset(s3)
print('v2 non-clean problems',len(f2),'v3',len(f3))
print('v2-only:',sorted(f2-f3))
print('v3-only:',sorted(f3-f2))
print('both:',len(f2&f3))
print()
# workload-level asymmetry
w2={(pk,u) for pk,d in s2.items() for u,x in d.items() if x!='PASSED'}
w3={(pk,u) for pk,d in s3.items() for u,x in d.items() if x!='PASSED'}
print('workload non-pass: v2',len(w2),'v3',len(w3),'both',len(w2&w3),'v2only',len(w2-w3),'v3only',len(w3-w2))
c=Counter(pk for pk,u in (w3-w2)); print('v3-only workloads by problem (top):',c.most_common(15))
c=Counter(pk for pk,u in (w2-w3)); print('v2-only workloads by problem:',c.most_common(15))
print()
# DB comparison
con=sqlite3.connect(ROOT+'/leaderboard/solbench.db')
dbf={}
for sid in (2,3):
    dbf[sid]={pk for (pk,) in con.execute("select distinct problem_key from result where submission_id=? and status='FAILED'",(sid,))}
    print('DB sub',sid,'problems with >=1 FAILED row:',len(dbf[sid]))
print('DB v2-only:',sorted(dbf[2]-dbf[3]))
print('DB v3-only:',sorted(dbf[3]-dbf[2]))
# does DB FAILED problem set == artifact non-clean set?
print('DB2 set == artifact f2 ?', dbf[2]==f2, len(dbf[2]^f2), sorted(dbf[2]^f2))
print('DB3 set == artifact f3 ?', dbf[3]==f3, len(dbf[3]^f3), sorted(dbf[3]^f3))
# problems with zero rows in DB
allp={pk for pk in s2}
for sid in (1,2,3,4):
    have={pk for (pk,) in con.execute("select distinct problem_key from result where submission_id=?",(sid,))}
    print('sub',sid,'problems with no row at all:',sorted(allp-have))
