"""Roofline + config sweep. What bandwidth is actually reachable, and at which config?"""
import time
import torch, triton
import triton.language as tl

def wall(fn, iters=300):
    for _ in range(30): fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/iters*1000

# ---------- roofline: pure 2:3 read / 1:3 write streaming copy of same volume ----------
@triton.jit
def _copy(src, dst, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid*BLOCK + tl.arange(0, BLOCK)
    m = o < N
    tl.store(dst+o, tl.load(src+o, mask=m), mask=m)

print("=== roofline: plain copy (read N + write N) ===")
for B,S in [(4,4096),(16,1024),(32,512),(4,2048)]:
    N = B*768*S
    a = torch.randn(N, device="cuda"); b = torch.empty(N, device="cuda")
    t = wall(lambda: _copy[(triton.cdiv(N,4096),)](a,b,N,BLOCK=4096))
    print(f"  B={B:3d} S={S:5d} N={N/1e6:7.2f}M  {t:8.2f}us  {2*N*4/(t*1e-6)/1e9:7.0f} GB/s")

# ---------- config sweep on the fused kernel ----------
import kernel as K

print()
print("=== fused kernel config sweep (fp64 accum) ===")
D = 256
for B,S in [(4,4096),(16,1024),(32,512),(4,2048),(64,128),(1,512)]:
    u = torch.randn((B,768,S), device="cuda"); w = torch.randn((768,1,3), device="cuda"); bi = torch.randn((768,), device="cuda")
    DS = D*S
    out = torch.empty((3,B,D,S), device="cuda")
    best = None
    res = []
    for blk in [64,128,256,512,1024,2048]:
        if blk > max(S, 64)*2: continue
        for nw in [1,2,4,8]:
            nsb = triton.cdiv(S, blk); even = (S % blk)==0
            try:
                f = lambda: K._hyena_fused[(nsb,D,B)](u,w,bi,out,S,DS,B*DS,3*DS,0,
                        BLOCK_S=blk,EVEN=even,num_warps=nw,num_stages=1)
                f()
                torch.cuda.synchronize()
                t = wall(f)
            except Exception as e:
                continue
            gbs = 2*B*768*S*4/(t*1e-6)/1e9
            res.append((t, blk, nw, gbs))
    res.sort()
    cur_blk,cur_nw,_,_ = K._cfg(S)
    print(f"  B={B:3d} S={S:5d}  (current blk={cur_blk} nw={cur_nw})")
    for t,blk,nw,gbs in res[:4]:
        print(f"      blk={blk:5d} nw={nw}  {t:8.2f}us  {gbs:7.0f} GB/s")
