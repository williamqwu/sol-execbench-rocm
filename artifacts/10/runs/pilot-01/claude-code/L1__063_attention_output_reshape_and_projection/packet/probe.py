"""How much of the gap is my A-addressing vs Triton GEMM in general?

Compares, at each shape:
  ref         : torch transpose+contiguous copy, then rocBLAS matmul
  copy+blas   : explicit copy then rocBLAS (same thing, decomposed)
  blas_only   : rocBLAS on an already-contiguous A  (== the SOL floor for BLAS)
  triton_ctg  : plain Triton GEMM on contiguous A  (Triton's own ceiling)
"""
import time, json, torch, triton
import triton.language as tl

dev = "cuda:0"
H, D, N = 128, 128, 7168
K = H * D


@triton.jit
def _gemm(A, B, C, M, N, stride_am, stride_bn, stride_cm,
          K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          GM: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    nig = GM * npn
    gid = pid // nig
    fpm = gid * GM
    gsm = min(npm - fpm, GM)
    pid_m = fpm + ((pid % nig) % gsm)
    pid_n = (pid % nig) // gsm
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    ap = A + om[:, None] * stride_am + ok[None, :]
    bp = B + ok[:, None] + on[None, :] * stride_bn
    acc = tl.zeros((BM, BN), tl.float32)
    mm = om < M
    for _ in range(K // BK):
        a = tl.load(ap, mask=mm[:, None], other=0.)
        b = tl.load(bp)
        acc = tl.dot(a, b, acc)
        ap += BK
        bp += BK
    tl.store(C + om[:, None] * stride_cm + on[None, :], acc.to(C.dtype.element_ty),
             mask=mm[:, None])


def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


torch.manual_seed(0)
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
wt = w.t().contiguous()   # [K, N] for the triton kernel, N-major
shapes = [(1, 131), (1, 2048), (4, 512), (8, 512), (32, 128), (2, 1024)]
print(f"{'B':>3}{'S':>6}{'M':>7} {'ref':>8} {'blasonly':>9} {'tri_ctg':>9} {'copy':>8}")
for B, S in shapes:
    a = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    M = B * S
    ac = a.transpose(1, 2).reshape(B, S, K).contiguous().reshape(M, K)
    t_ref = bench(lambda: torch.matmul(a.transpose(1, 2).reshape(B, S, K), w.t()))
    t_blas = bench(lambda: torch.matmul(ac, w.t()))
    t_copy = bench(lambda: a.transpose(1, 2).reshape(B, S, K).contiguous())
    c = torch.empty((M, N), device=dev, dtype=torch.bfloat16)
    BM, BN, BK, GM, nw = (256, 256, 64, 8, 8) if M >= 512 else (64, 64, 128, 8, 4)
    g = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    f = lambda: _gemm[g](ac, w, c, M, N, K, K, N, K=K, BM=BM, BN=BN, BK=BK, GM=GM,
                         num_warps=nw, num_stages=2)
    f()
    err = (c.float() - torch.matmul(ac, w.t()).float()).abs().max().item()
    t_tri = bench(f)
    print(f"{B:>3}{S:>6}{M:>7} {t_ref:8.3f} {t_blas:9.3f} {t_tri:9.3f} {t_copy:8.3f}  err={err:.3f}")
    del a, ac, c
    torch.cuda.empty_cache()
