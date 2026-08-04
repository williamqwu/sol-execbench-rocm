import torch, triton, triton.language as tl, json

N, K = 5120, 2048
dev = "cuda:0"
B = torch.randn(N, K, device=dev, dtype=torch.float16)


def bench(fn, iters=100, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    ts = []
    for _ in range(5):
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) / iters * 1e3)
    return min(ts)


@triton.jit
def empty_k(x):
    pass


@triton.jit
def read_k(Bp, Out, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BN, BK), dtype=tl.float32)
    for k in range(0, K, BK):
        b = tl.load(Bp + (pid * BN + tl.arange(0, BN))[:, None] * K + (k + tl.arange(0, BK))[None, :])
        acc += b.to(tl.float32)
    s = tl.sum(tl.sum(acc, 1), 0)
    tl.store(Out + pid, s)


print("empty launch us:", bench(lambda: empty_k[(1,)](B)))

for BN in (16, 32, 64):
    out = torch.zeros(N // BN, device=dev, dtype=torch.float32)
    for BK in (64, 128, 256):
        t = bench(lambda: read_k[(N // BN,)](B, out, K, BN, BK, num_warps=4))
        print(f"read BN={BN} BK={BK}: {t:.2f} us  {N*K*2/t*1e-3:.0f} GB/s")

# torch read floor
print("B.sum():", bench(lambda: B.sum()), "us")
Bf = B.float()
print("B.clone():", bench(lambda: B.clone()), "us")

# F.linear vs matmul at small M
import torch.nn.functional as F
for M in (1, 8, 16, 64, 128):
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    Bt = B.T.contiguous()
    print(M, "matmul", round(bench(lambda: torch.matmul(A, B.T)), 2),
          "linear", round(bench(lambda: F.linear(A, B)), 2),
          "mm_NN", round(bench(lambda: torch.matmul(A, Bt)), 2))
