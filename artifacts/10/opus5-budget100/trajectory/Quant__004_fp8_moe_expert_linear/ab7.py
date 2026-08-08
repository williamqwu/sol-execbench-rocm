import sys, torch, triton
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import tk6 as tk5, tk7 as tk6
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
for M in [384, 1024, 1536, 2048, 2432, 3072, 4096]:
    hs = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    rw = torch.randn(M, 1, device=dev, dtype=torch.bfloat16)
    t5 = bench(lambda: tk5.moe(hs, rw, guw, dw))
    t6 = bench(lambda: tk6.moe(hs, rw, guw, dw))
    print(f"M={M:5d} tk7 {t5*1e3:7.1f}us  tk6 {t6*1e3:7.1f}us  "
          f"{'+' if t6 < t5 else '-'}{abs(t5-t6)/t5*100:.1f}%  "
          f"cfg1={tk6._cfg1(M, I)} cfg2={tk6._cfg2(M, H)}", flush=True)
