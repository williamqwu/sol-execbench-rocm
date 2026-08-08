import torch, time, triton, triton.language as tl
dev = 'cuda'


def bench(fn, n=200, w=50):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


N = 1 << 26  # 64M bf16 = 128MB
x = torch.randn(N, device=dev, dtype=torch.bfloat16)
y = torch.empty(N, device=dev, dtype=torch.bfloat16)
t = bench(lambda: y.copy_(x))
print(f"copy 128MB->128MB: {t*1e3:.1f} us  {2*N*2/(t/1e3)/1e12:.2f} TB/s")


@triton.jit
def rd(X, O, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + i, mask=i < N, other=0.0)
    s = tl.sum(v.to(tl.float32))
    tl.store(O + tl.program_id(0), s)


for B in [1024, 2048, 4096, 8192]:
    o = torch.empty(triton.cdiv(N, B), device=dev)
    t = bench(lambda: rd[(triton.cdiv(N, B),)](x, o, N, BLOCK=B, num_warps=4))
    print(f"read-only B={B}: {t*1e3:.1f} us  {N*2/(t/1e3)/1e12:.2f} TB/s")
