import torch, triton
import triton.language as tl

@triton.jit
def _build2(idx_ptr, cnt_ptr, slots_ptr, ovf_ptr, novf_ptr, M, C: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < M
    t = tl.load(idx_ptr + offs, mask=m, other=0).to(tl.int32)
    p = tl.atomic_add(cnt_ptr + t, 1, mask=m)
    tl.store(slots_ptr + t * C + p, offs.to(tl.int32), mask=m & (p < C))
    bad = m & (p >= C)
    q = tl.atomic_add(novf_ptr + tl.zeros([BLOCK], tl.int32), 1, mask=bad)
    tl.store(ovf_ptr + 2 * q, t, mask=bad)
    tl.store(ovf_ptr + 2 * q + 1, offs.to(tl.int32), mask=bad)

# scalar-k gather: accumulate a single BLOCK_H vector, one row per iteration
@triton.jit
def _gath_s(final_ptr, exp_ptr, cnt_ptr, slots_ptr, ovf_ptr, novf_ptr, out_ptr,
            H: tl.constexpr, C: tl.constexpr, BLOCK_H: tl.constexpr, U: tl.constexpr):
    t = tl.program_id(0)
    ph = tl.program_id(1)
    h = ph * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = h < H
    base = t.to(tl.int64) * H + h
    acc = tl.load(final_ptr + base, mask=hm, other=0.0).to(tl.float32)
    cnt = tl.load(cnt_ptr + t)
    n = tl.minimum(cnt, C)
    sp = slots_ptr + t * C
    for k in range(0, n, U):
        for u in tl.static_range(U):
            kk = k + u
            r = tl.load(sp + kk, mask=kk < n, other=-1)
            v = tl.load(exp_ptr + r.to(tl.int64) * H + h, mask=hm & (kk < n) & (r >= 0), other=0.0)
            acc += v.to(tl.float32)
    nov = tl.load(novf_ptr)
    for j in range(0, nov):
        ot = tl.load(ovf_ptr + 2 * j)
        if ot == t:
            orow = tl.load(ovf_ptr + 2 * j + 1)
            acc += tl.load(exp_ptr + orow.to(tl.int64) * H + h, mask=hm, other=0.0).to(tl.float32)
    tl.store(out_ptr + base, acc.to(tl.bfloat16), mask=hm)

def run2(f, e, i, BLOCK_H=1024, U=4, C=32, BB=256, nw=4, bnw=4, gwait=1):
    N, H = f.shape; M = i.shape[0]; dev = f.device
    buf = torch.zeros(N + 1, dtype=torch.int32, device=dev)
    slots = torch.empty(N * C, dtype=torch.int32, device=dev)
    ovf = torch.empty(2 * M, dtype=torch.int32, device=dev)
    out = torch.empty_like(f)
    _build2[(triton.cdiv(M, BB),)](i, buf, slots, ovf, buf[N:], M, C=C, BLOCK=BB, num_warps=bnw)
    NH = triton.cdiv(H, BLOCK_H)
    _gath_s[(N, NH)](f, e, buf, slots, ovf, buf[N:], out, H=H, C=C, BLOCK_H=BLOCK_H, U=U,
                     num_warps=nw, waves_per_eu=gwait)
    return out

# atomic variant
@triton.jit
def _scat(exp_ptr, idx_ptr, out_ptr, M, H: tl.constexpr, BLOCK_H: tl.constexpr):
    r = tl.program_id(0); ph = tl.program_id(1)
    h = ph * BLOCK_H + tl.arange(0, BLOCK_H); hm = h < H
    t = tl.load(idx_ptr + r)
    v = tl.load(exp_ptr + r.to(tl.int64) * H + h, mask=hm, other=0.0)
    tl.atomic_add(out_ptr + t.to(tl.int64) * H + h, v, mask=hm, sem="relaxed")

def run_at(f, e, i, BLOCK_H=1024, nw=4):
    N, H = f.shape; M = i.shape[0]
    out = f.clone()
    NH = triton.cdiv(H, BLOCK_H)
    _scat[(M, NH)](e, i, out, M, H=H, BLOCK_H=BLOCK_H, num_warps=nw)
    return out
