import torch, triton, triton.language as tl, traceback
N, K = 5120, 2048
dev = "cuda:0"
torch.manual_seed(0)
Bm = torch.randn(N, K, device=dev, dtype=torch.float16)

@triton.jit
def gemv(Ap, Bp, Cp, M, N: tl.constexpr, K: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pn = tl.program_id(0)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    Bptr = Bp + rn[:, None]*K + rk[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    koff = 0
    for _ in tl.range(0, K // BK):
        b = tl.load(Bptr).to(tl.float32)
        for m in tl.static_range(BM):
            a = tl.load(Ap + m*K + koff + rk, mask=(m < M), other=0.0).to(tl.float32)
            p = tl.sum(a[None, :] * b, axis=1)
            acc += tl.where((tl.arange(0, BM) == m)[:, None], p[None, :], 0.0)
        Bptr += BK
        koff += BK
    rm = tl.arange(0, BM)
    tl.store(Cp + rm[:, None]*N + rn[None, :], acc.to(tl.float16), mask=rm[:, None] < M)

M = 1
A = torch.randn(M, K, device=dev, dtype=torch.float16)
C = torch.empty(M, N, device=dev, dtype=torch.float16)
try:
    gemv[(N//32,)](A, Bm, C, M, N, K, 1, 32, 128, num_warps=4, num_stages=2)
    torch.cuda.synchronize()
    ref = torch.matmul(A, Bm.T)
    print("max err", (C.float()-ref.float()).abs().max().item())
    print("C", C[0,:5], "ref", ref[0,:5])
except Exception:
    traceback.print_exc()
