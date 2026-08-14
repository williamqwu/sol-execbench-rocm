"""Per-(B,S) config sweep for v2, plus a generalizing fallback heuristic."""
import json
import torch, triton
import k2

D = 256
SHAPES = []
for line in open("workload.jsonl"):
    if line.strip():
        a = json.loads(line)["axes"]; SHAPES.append((a["batch_size"], a["seq_len"]))

def wall(fn, iters=500):
    for _ in range(60): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Event(True); en=torch.cuda.Event(True)
    st.record()
    for _ in range(iters): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/iters*1000

out_cfg = {}
tot = 0.0
print(f"{'B':>4s}{'S':>7s}  {'best':>9s} {'blk':>6s}{'nw':>4s}{'ns':>3s} {'GB/s':>7s}   runners-up")
for (B,S) in SHAPES:
    u=torch.randn((B,768,S),device="cuda"); w=torch.randn((768,1,3),device="cuda"); bi=torch.randn((768,),device="cuda")
    TDS=B*D*S
    res=[]
    for blk in [64,128,256,512,1024,2048,4096]:
        if blk > 2*max(S,64): continue
        nsb=triton.cdiv(S,blk); even=(S%blk)==0
        for nw in [1,2,4,8,16]:
            for ns in [1,2]:
                def f(blk=blk,nw=nw,ns=ns,nsb=nsb,even=even):
                    o=torch.empty((3,B,D,S),device="cuda")
                    k2._fused_v2[(nsb,D,B)](u,w,bi,o,TDS,S=S,BLOCK_S=blk,NSB=nsb,
                        EVEN=even,BRANCH=False,num_warps=nw,num_stages=ns)
                    return o.unbind(0)
                try:
                    f(); torch.cuda.synchronize(); t=wall(f)
                except Exception:
                    continue
                res.append((t,blk,nw,ns))
    res.sort()
    t,blk,nw,ns = res[0]
    out_cfg[(B,S)] = (blk,nw,ns)
    tot += t
    ru = " ".join(f"{b}/{w_}/{n}:{tt:.1f}" for tt,b,w_,n in res[1:4])
    print(f"{B:4d}{S:7d}  {t:8.2f}u {blk:6d}{nw:4d}{ns:3d} {2*B*768*S*4/(t*1e-6)/1e9:7.0f}   {ru}")

print(f"\nTOTAL best-per-shape = {tot:.2f}us")
print("\nCFG = {")
for k in sorted(out_cfg, key=lambda x:(x[1],x[0])):
    print(f"    {k}: {out_cfg[k]},")
print("}")
