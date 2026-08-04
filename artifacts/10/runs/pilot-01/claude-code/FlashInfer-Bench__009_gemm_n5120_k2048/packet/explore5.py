import torch, triton, triton.language as tl
N, K = 5120, 2048
dev = "cuda:0"
Bm = torch.randn(N, K, device=dev, dtype=torch.float16)

def gpu_time(fn, iters=100):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters): fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True); ts=[]
    for _ in range(5):
        st.record(); g.replay(); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en)/iters*1e3)
    return min(ts)

@triton.jit
def nop(X):
    pass

d = torch.zeros(1, device=dev)
for gsz in (1, 64, 256, 1024):
    print(f"empty kernel grid={gsz}: {gpu_time(lambda: nop[(gsz,)](d)):.2f} us")

# pure streaming read of B, one row-block per program, vectorized
@triton.jit
def rd(Bp, Out, K: tl.constexpr, ROWS: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * ROWS * K + tl.arange(0, ROWS * BK)
    acc = tl.zeros((ROWS * BK,), dtype=tl.float32)
    for _ in tl.range(0, (ROWS * K) // (ROWS * BK)):
        acc += tl.load(Bp + off).to(tl.float32)
        off += ROWS * BK
    tl.store(Out + pid, tl.sum(acc))

print("\npure read of B (21MB):")
for ROWS in (2, 4, 8, 16, 20, 32):
    nblk = N // ROWS
    out = torch.zeros(nblk, device=dev, dtype=torch.float32)
    for BK in (256, 512, 1024):
        if (ROWS * K) % (ROWS * BK): continue
        for nw in (4, 8):
            try:
                t = gpu_time(lambda: rd[(nblk,)](Bm, out, K, ROWS, BK, num_warps=nw))
                print(f"  ROWS={ROWS:>2} blocks={nblk:>4} BK={BK:>4} w{nw}: {t:6.2f} us  {N*K*2/t*1e-3:5.0f} GB/s")
            except Exception as e:
                pass
