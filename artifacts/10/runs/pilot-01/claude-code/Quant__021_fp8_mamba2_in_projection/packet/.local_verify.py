"""Local stand-in for ./verify (harness reference needs py3.12 StrEnum; box is 3.10)."""
import enum
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self): return str(self.value)
    enum.StrEnum = StrEnum

import json, sys, time, importlib
sys.path.insert(0, ".")
import torch
import reference
import kernel
importlib.reload(kernel)

dev = torch.device("cuda:0")
K, N = 2688, 13952

wls = [json.loads(l) for l in open("workload.jsonl") if l.strip()]

def bench(fn, iters=20):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts)//2] * 1e3

allpass = True
tot_r = tot_m = 0.0
for i, wl in enumerate(wls):
    M = wl["axes"]["M"]
    tol = wl["tolerance"]
    torch.manual_seed(1234 + i)
    ins = reference.get_inputs({"M": M, "hidden_size": K, "projection_size": N}, dev)
    x, w = ins["hidden_states"], ins["weight"]

    ref = reference.run(x, w)
    out = kernel.run(x, w)

    ok_shape = (out.shape == ref.shape and out.dtype == ref.dtype)
    rf, of = ref.float(), out.float()
    d = (of - rf).abs()
    thr = tol["max_atol"] + tol["max_rtol"] * rf.abs()
    ratio = (d <= thr).float().mean().item()
    passed = ok_shape and ratio >= tol["required_matched_ratio"]
    exact = torch.equal(out, ref)

    tr = bench(lambda: reference.run(x, w), 10)
    tm = bench(lambda: kernel.run(x, w))
    tot_r += tr; tot_m += tm
    allpass &= passed
    print(f"  M={M:<6} {'PASS' if passed else 'FAIL'}  ratio={ratio:.6f} "
          f"maxerr={d.max().item():.4f}  exact={exact}  "
          f"ref={tr:8.3f}ms  mine={tm:7.3f}ms  {tr/tm:6.2f}x")

print(f"\n{'ALL PASS' if allpass else 'FAILURES'}   total ref={tot_r:.1f}ms mine={tot_m:.1f}ms  {tot_r/tot_m:.2f}x")
sys.exit(0 if allpass else 1)
