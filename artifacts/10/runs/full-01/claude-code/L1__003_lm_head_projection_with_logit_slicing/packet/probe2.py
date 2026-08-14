import torch, importlib
import triton, triton.language as tl

dev = torch.device("cuda:0")

def timeit(fn, iters=20, warmup=5, reps=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        st = torch.cuda.Event(True); en = torch.cuda.Event(True)
        st.record()
        for _ in range(iters): fn()
        en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) / iters * 1e3)
    return min(ts)

# --- pure streaming read bandwidth with a proper triton kernel
@triton.jit
def _rd(X, OUT, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + o, mask=o < n, other=0.0)
    s = tl.sum(v)
    if s == 1234.5678:
        tl.store(OUT + pid, s)

print("=== streaming read bw ===")
nb = 512 * 1024 * 1024
x = torch.randn(nb // 2, dtype=torch.bfloat16, device=dev)
out = torch.empty(65536, dtype=torch.float32, device=dev)
for BLOCK in (2048, 4096, 8192):
    g = (triton.cdiv(x.numel(), BLOCK),)
    t = timeit(lambda: _rd[g](x, out, x.numel(), BLOCK=BLOCK, num_warps=8))
    print(f"  BLOCK={BLOCK}: {t:8.1f}us -> {nb/t*1e-6:6.2f} TB/s")
del x; torch.cuda.empty_cache()

# --- libraries
print("=== libs ===")
for m in ["aiter", "hipblaslt", "ck4inductor", "flash_attn"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m}: OK {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m}: NO ({type(e).__name__})")

# --- 2D vs 3D matmul, and out= variant
print("=== torch matmul variants ===")
H, V = 2048, 102400
w = torch.randn(V, H, dtype=torch.bfloat16, device=dev)
wt = w.t()
for B, S in [(1, 128), (32, 128), (64, 128), (1, 2048), (2, 2048)]:
    M = B * S
    x3 = torch.randn(B, S, H, dtype=torch.bfloat16, device=dev)
    x2 = x3.reshape(M, H)
    o2 = torch.empty(M, V, dtype=torch.bfloat16, device=dev)
    t3 = timeit(lambda: torch.matmul(x3, wt), iters=10)
    t2 = timeit(lambda: torch.matmul(x2, wt), iters=10)
    to = timeit(lambda: torch.mm(x2, wt, out=o2), iters=10)
    tl_ = timeit(lambda: torch.nn.functional.linear(x2, w), iters=10)
    print(f"  B={B:3d} S={S:5d} M={M:5d}: 3d={t3:8.1f} 2d={t2:8.1f} out={to:8.1f} linear={tl_:8.1f}")
    del x3, x2, o2; torch.cuda.empty_cache()
