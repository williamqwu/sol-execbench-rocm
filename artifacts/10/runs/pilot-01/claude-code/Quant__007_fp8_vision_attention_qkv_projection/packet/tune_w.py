import torch, triton, triton.language as tl
from triton.testing import do_bench

dev = "cuda:0"
torch.manual_seed(0)
N, Kd = 4608, 1536
w = torch.randn(N, Kd, dtype=torch.bfloat16, device=dev) * 0.05
qw = torch.empty((N, Kd), dtype=torch.float8_e4m3fn, device=dev)
sw = torch.empty((N // 128, Kd // 128), dtype=torch.float32, device=dev)

RECIP = tl.constexpr(1.0 / 448.0)
EMAX = tl.constexpr(448.0)


# pure bandwidth floor: bf16 -> fp8 elementwise copy
@triton.jit
def _copy(w_ptr, o_ptr, n_elem, BLK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLK + tl.arange(0, BLK)
    m = o < n_elem
    tl.store(o_ptr + o, tl.load(w_ptr + o, mask=m).to(tl.float8e4nv), mask=m)


# variant A: one program per 128x128 tile (current)
@triton.jit
def _qa(w_ptr, qw_ptr, sw_ptr, sn, sk):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    on = pid_n * 128 + tl.arange(0, 128)
    ok = pid_k * 128 + tl.arange(0, 128)
    x = tl.load(w_ptr + on[:, None] * sn + ok[None, :]).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x)) * RECIP, 1e-12)
    q = tl.minimum(tl.maximum(x / s, -EMAX), EMAX)
    tl.store(qw_ptr + on[:, None] * sn + ok[None, :], q.to(tl.float8e4nv))
    tl.store(sw_ptr + pid_n * sk + pid_k, s)


# variant B: one program per row-block, loops over all K tiles (better reuse of ptrs)
@triton.jit
def _qb(w_ptr, qw_ptr, sw_ptr, sn, sk, NKB: tl.constexpr):
    pid_n = tl.program_id(0)
    on = pid_n * 128 + tl.arange(0, 128)
    ok = tl.arange(0, 128)
    for kb in tl.range(0, NKB):
        p = w_ptr + on[:, None] * sn + (kb * 128 + ok)[None, :]
        x = tl.load(p).to(tl.float32)
        s = tl.maximum(tl.max(tl.abs(x)) * RECIP, 1e-12)
        q = tl.minimum(tl.maximum(x / s, -EMAX), EMAX)
        tl.store(qw_ptr + on[:, None] * sn + (kb * 128 + ok)[None, :], q.to(tl.float8e4nv))
        tl.store(sw_ptr + pid_n * sk + kb, s)


nel = N * Kd
for blk in [2048, 4096, 8192]:
    t = do_bench(lambda: _copy[(triton.cdiv(nel, blk),)](w, qw, nel, BLK=blk),
                 warmup=50, rep=200) * 1e3
    print(f"copy floor BLK={blk:5d}: {t:6.2f} us")

for nw in [1, 2, 4, 8, 16]:
    for ns in [1, 2]:
        try:
            t = do_bench(lambda: _qa[(N // 128, Kd // 128)](
                w, qw, sw, w.stride(0), sw.stride(0), num_warps=nw, num_stages=ns),
                warmup=50, rep=200) * 1e3
            print(f"A nw={nw:2d} ns={ns}: {t:6.2f} us")
        except Exception as e:
            print("A", nw, ns, "err", str(e)[:60])

for nw in [4, 8, 16]:
    for ns in [1, 2]:
        try:
            t = do_bench(lambda: _qb[(N // 128,)](
                w, qw, sw, w.stride(0), sw.stride(0), NKB=Kd // 128,
                num_warps=nw, num_stages=ns), warmup=50, rep=200) * 1e3
            print(f"B nw={nw:2d} ns={ns}: {t:6.2f} us")
        except Exception as e:
            print("B", nw, ns, "err", str(e)[:60])
