"""Full end-to-end config sweep over all 16 workload shapes (alloc+launch+unbind included)."""
import json, time
import torch, triton
import triton.language as tl
import kernel as K

D = 256
SHAPES = []
for line in open("workload.jsonl"):
    if line.strip():
        a = json.loads(line)["axes"]
        SHAPES.append((a["batch_size"], a["seq_len"]))

def wall(fn, iters=400):
    for _ in range(50): fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/iters*1000

best_cfg = {}
print(f"{'B':>4s}{'S':>7s}  {'best':>9s} {'blk':>6s}{'nw':>4s}{'ns':>3s}  {'GB/s':>7s}   {'cur':>8s}")
for (B, S) in SHAPES:
    u = torch.randn((B,768,S), device="cuda")
    w = torch.randn((768,1,3), device="cuda")
    bi = torch.randn((768,), device="cuda")
    DS = D*S
    res = []
    for blk in [64,128,256,512,1024,2048,4096]:
        nsb = triton.cdiv(S, blk); even = (S % blk) == 0
        if blk > 2*max(S,64): continue
        for nw in [1,2,4,8]:
            for ns in [1,2]:
                def f(blk=blk, nw=nw, ns=ns, nsb=nsb, even=even):
                    o = torch.empty((3,B,D,S), device="cuda")
                    K._hyena_fused[(nsb,D,B)](u,w,bi,o,S,DS,B*DS,3*DS,0,
                        BLOCK_S=blk,EVEN=even,num_warps=nw,num_stages=ns)
                    return o.unbind(0)
                try:
                    f(); torch.cuda.synchronize()
                    t = wall(f)
                except Exception:
                    continue
                res.append((t, blk, nw, ns))
    res.sort()
    t,blk,nw,ns = res[0]
    best_cfg[S] = (blk,nw,ns)
    gbs = 2*B*768*S*4/(t*1e-6)/1e9
    tcur = wall(lambda: K.run(u,w,bi))
    print(f"{B:4d}{S:7d}  {t:8.2f}us {blk:6d}{nw:4d}{ns:3d}  {gbs:7.0f}   {tcur:7.2f}us")

print()
print("per-S best (blk, nw, ns):")
for S in sorted(best_cfg):
    print(f"  {S:5d}: {best_cfg[S]}")
