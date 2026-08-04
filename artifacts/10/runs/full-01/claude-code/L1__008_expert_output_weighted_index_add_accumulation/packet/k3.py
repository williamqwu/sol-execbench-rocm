import torch, triton
import triton.language as tl

@triton.jit
def _copy(src, dst, n, BLOCK: tl.constexpr):
    p = tl.program_id(0); o = p * BLOCK + tl.arange(0, BLOCK); m = o < n
    tl.store(dst + o, tl.load(src + o, mask=m, cache_modifier=".cg"), mask=m, cache_modifier=".cg")

@triton.jit
def _scat(exp_ptr, idx_ptr, out_ptr, M, H: tl.constexpr, BLOCK_H: tl.constexpr,
          RPB: tl.constexpr, NH: tl.constexpr):
    pid = tl.program_id(0)
    ph = pid % NH
    rb = pid // NH
    h = ph * BLOCK_H + tl.arange(0, BLOCK_H); hm = h < H
    for u in tl.static_range(RPB):
        r = rb * RPB + u
        if r < M:
            t = tl.load(idx_ptr + r)
            v = tl.load(exp_ptr + r.to(tl.int64) * H + h, mask=hm, other=0.0, cache_modifier=".cg")
            tl.atomic_add(out_ptr + t.to(tl.int64) * H + h, v, mask=hm, sem="relaxed")

def run_at(f, e, i, BLOCK_H=1024, nw=4, RPB=1, CB=4096, cnw=8):
    N, H = f.shape; M = i.shape[0]
    out = torch.empty_like(f)
    ne = N * H
    _copy[(triton.cdiv(ne, CB),)](f, out, ne, BLOCK=CB, num_warps=cnw)
    NH = triton.cdiv(H, BLOCK_H)
    _scat[(triton.cdiv(M, RPB) * NH,)](e, i, out, M, H=H, BLOCK_H=BLOCK_H, RPB=RPB, NH=NH, num_warps=nw)
    return out
