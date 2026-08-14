"""Compare v1 vs v2 (constexpr S, branch) vs fp32 speed-ceiling, across all shapes."""
import json
import torch, triton
import k2, kernel as K

D = 256
SHAPES = []
for line in open("workload.jsonl"):
    if line.strip():
        a = json.loads(line)["axes"]; SHAPES.append((a["batch_size"], a["seq_len"]))

BEST = {128:(64,2,1),256:(128,4,1),293:(64,1,1),512:(512,4,1),
        1024:(256,4,1),2048:(1024,2,1),4096:(128,2,1)}

def wall(fn, iters=400):
    for _ in range(50): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Event(True); en=torch.cuda.Event(True)
    st.record()
    for _ in range(iters): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/iters*1000

tot = {"v1":0.,"v2":0.,"v2b":0.,"f32":0.}
print(f"{'B':>4s}{'S':>7s} {'v1':>9s} {'v2':>9s} {'v2+br':>9s} {'fp32ceil':>9s}  {'best blk/nw':>12s}")
for (B,S) in SHAPES:
    u=torch.randn((B,768,S),device="cuda"); w=torch.randn((768,1,3),device="cuda"); bi=torch.randn((768,),device="cuda")
    blk,nw,ns = BEST[S]
    nsb=triton.cdiv(S,blk); even=(S%blk)==0
    TDS=B*D*S

    t1 = wall(lambda: K.run(u,w,bi))

    def f2(BR):
        o=torch.empty((3,B,D,S),device="cuda")
        k2._fused_v2[(nsb,D,B)](u,w,bi,o,TDS,S=S,BLOCK_S=blk,NSB=nsb,EVEN=even,BRANCH=BR,num_warps=nw,num_stages=ns)
        return o.unbind(0)
    f2(False); torch.cuda.synchronize(); t2 = wall(lambda: f2(False))
    f2(True); torch.cuda.synchronize(); t2b = wall(lambda: f2(True))

    def f32():
        o=torch.empty((3,B,D,S),device="cuda")
        k2._fused_f32[(nsb,D,B)](u,w,bi,o,TDS,S=S,BLOCK_S=blk,NSB=nsb,EVEN=even,BRANCH=False,num_warps=nw,num_stages=ns)
        return o.unbind(0)
    f32(); torch.cuda.synchronize(); t3 = wall(f32)

    # correctness of v2 vs v1
    a = K.run(u,w,bi); b = f2(False); torch.cuda.synchronize()
    okv2 = all(torch.equal(x,y) for x,y in zip(a,b))
    c = f2(True); torch.cuda.synchronize()
    okv2b = all(torch.equal(x,y) for x,y in zip(a,c))

    tot["v1"]+=t1; tot["v2"]+=t2; tot["v2b"]+=t2b; tot["f32"]+=t3
    print(f"{B:4d}{S:7d} {t1:8.2f}u {t2:8.2f}u {t2b:8.2f}u {t3:8.2f}u  {blk:5d}/{nw}  "
          f"{'ok' if okv2 else 'MISMATCH'} {'ok' if okv2b else 'MISMATCH'}")

print()
for k,v in tot.items(): print(f"  total {k:5s} = {v:8.2f}us")
