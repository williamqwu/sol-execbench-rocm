import sys, torch, triton
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import tk6, tk7
dev = 'cuda'
H, I = 3584, 2048


def bench(fn, n=60, w=25):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in ev)
    return ts[len(ts) // 2]


guw = torch.randn(2 * I, H, device=dev, dtype=torch.bfloat16) * H ** -0.5
dw = torch.randn(H, I, device=dev, dtype=torch.bfloat16) * I ** -0.5
MS = [384, 640, 1024, 1536, 2048, 2432, 3072, 4096]
acc = {m: [[], []] for m in MS}
for rep in range(5):
    for M in MS:
        hs = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
        rw = torch.randn(M, 1, device=dev, dtype=torch.bfloat16)
        acc[M][0].append(bench(lambda: tk6.moe(hs, rw, guw, dw)))
        acc[M][1].append(bench(lambda: tk7.moe(hs, rw, guw, dw)))
        acc[M][1].append(bench(lambda: tk7.moe(hs, rw, guw, dw)))
        acc[M][0].append(bench(lambda: tk6.moe(hs, rw, guw, dw)))
g6 = g7 = 1.0
for M in MS:
    a = sorted(acc[M][0])[len(acc[M][0]) // 2]
    b = sorted(acc[M][1])[len(acc[M][1]) // 2]
    g6 *= a
    g7 *= b
    print(f"M={M:5d} tk6 {a*1e3:7.1f}us  tk7(fused) {b*1e3:7.1f}us  "
          f"{(a-b)/a*100:+5.1f}%", flush=True)
n = len(MS)
print(f"geomean tk6 {g6**(1/n)*1e3:.1f}us  tk7 {g7**(1/n)*1e3:.1f}us  "
      f"{(g6**(1/n)-g7**(1/n))/g6**(1/n)*100:+.1f}%")
