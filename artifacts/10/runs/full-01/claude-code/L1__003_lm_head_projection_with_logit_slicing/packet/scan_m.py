import torch
H, V = 2048, 102400
dev = torch.device("cuda:0")
torch.manual_seed(0)
w = (torch.randn(V, H, dtype=torch.bfloat16, device=dev) * (1.0 / H**0.5))
wt = w.t()

def timeit(fn, iters=8, warmup=3, reps=3):
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

WL = [128, 256, 293, 691, 1024, 2048, 3011, 3412, 3988, 4096, 8192]
# scan around each workload M to find padding wins
cands = sorted(set(WL) | {
    64, 96, 160, 192, 224, 288, 320, 352, 384, 448, 512, 640, 704, 768, 896,
    960, 1088, 1152, 1280, 1536, 1792, 2560, 3072, 3136, 3200, 3264, 3328,
    3456, 3584, 4032, 4160, 4224, 6144, 8448,
})
res = {}
big = torch.randn(8448, H, dtype=torch.bfloat16, device=dev)
out = torch.empty(8448, V, dtype=torch.bfloat16, device=dev)
for M in cands:
    x = big[:M]
    o = out[:M]
    t = timeit(lambda: torch.mm(x, wt, out=o))
    res[M] = t
    star = " *" if M in WL else ""
    print(f"  M={M:5d}: {t:8.1f}us  {2*M*H*V/t*1e-6:7.1f}TF{star}", flush=True)

print("\n=== padding opportunities ===")
for M in WL:
    best = (res[M], M)
    for P, t in res.items():
        if P >= M and t < best[0]:
            best = (t, P)
    if best[1] != M:
        print(f"  M={M}: {res[M]:.1f}us -> pad {best[1]} {best[0]:.1f}us "
              f"({res[M]/best[0]:.2f}x)")
