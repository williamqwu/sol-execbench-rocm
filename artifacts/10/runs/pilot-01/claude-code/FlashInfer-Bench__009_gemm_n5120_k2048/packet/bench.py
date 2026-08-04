import json, sys, time
import torch

N, K = 5120, 2048
Ms = []
for line in open("workload.jsonl"):
    Ms.append(json.loads(line)["axes"]["M"])
Ms = sorted(set(Ms))
print("Ms:", Ms)

dev = "cuda:0"
torch.manual_seed(0)
B = torch.randn(N, K, device=dev, dtype=torch.float16)


def bench(fn, *args, iters=50, warmup=20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    # use events
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    ts = []
    for _ in range(5):
        st.record()
        for _ in range(iters):
            fn(*args)
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) / iters * 1e3)
    return min(ts)


def ref(A, B):
    return torch.matmul(A, B.T)


BYTES_B = N * K * 2
print(f"{'M':>7} {'torch us':>10} {'GB/s':>9} {'TFLOP/s':>9}")
res = {}
for M in Ms:
    A = torch.randn(M, K, device=dev, dtype=torch.float16)
    t = bench(ref, A, B)
    total_bytes = BYTES_B + M * K * 2 + M * N * 2
    flops = 2 * M * N * K
    res[M] = t
    print(f"{M:>7} {t:>10.1f} {total_bytes/t*1e-3:>9.1f} {flops/t*1e-6:>9.1f}")

json.dump(res, open("baseline.json", "w"))
