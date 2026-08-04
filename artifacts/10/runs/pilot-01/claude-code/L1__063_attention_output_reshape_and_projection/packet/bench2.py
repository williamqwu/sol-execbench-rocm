import json, time, torch, importlib, sys
import kernel

dev = "cuda:0"
H, D, N = 128, 128, 7168
K = H * D


def ref(a, w):
    b, h, s, d = a.shape
    return torch.matmul(a.transpose(1, 2).reshape(b, s, h * d), w.t())


def bench(fn, args, iters=30, warmup=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


shapes, tols = [], []
for l in open("workload.jsonl"):
    j = json.loads(l)
    shapes.append((j["axes"]["batch_size"], j["axes"]["seq_len"]))
    tols.append(j["tolerance"])

torch.manual_seed(0)
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
print(f"{'B':>3} {'S':>5} {'M':>6} {'ref ms':>9} {'ours ms':>9} {'x':>6}  {'maxerr':>10} {'ratio':>7} {'ok'}")
tot_r = tot_o = 0.0
for (B, S), tol in zip(shapes, tols):
    a = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    r = ref(a, w)
    o = kernel.run(a, w)
    assert o.shape == r.shape and o.dtype == r.dtype, (o.shape, o.dtype)
    rf, of = r.float(), o.float()
    diff = (rf - of).abs()
    allowed = tol["max_atol"] + tol["max_rtol"] * rf.abs()
    ratio = (diff <= allowed).float().mean().item()
    ok = ratio >= tol["required_matched_ratio"]
    t_r = bench(ref, (a, w))
    t_o = bench(kernel.run, (a, w))
    tot_r += t_r
    tot_o += t_o
    print(f"{B:>3} {S:>5} {B*S:>6} {t_r:9.3f} {t_o:9.3f} {t_r/t_o:6.2f}  "
          f"{diff.max().item():10.5f} {ratio:7.5f} {'OK' if ok else 'FAIL'}")
    del a, r, o, rf, of, diff, allowed
    torch.cuda.empty_cache()
print(f"total ref {tot_r:.3f} ms  ours {tot_o:.3f} ms  speedup {tot_r/tot_o:.2f}x")
