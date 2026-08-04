import json, sys, time
import enum

if not hasattr(enum, "StrEnum"):  # py3.10 shim, local harness only
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return self.value
    enum.StrEnum = StrEnum

import torch
import reference
import importlib

torch.manual_seed(0)
dev = "cuda:0"

seqs = [json.loads(l)["axes"]["seq_len"] for l in open("workload.jsonl")]
tols = [json.loads(l)["tolerance"] for l in open("workload.jsonl")]

import kernel
importlib.reload(kernel)


def make(sl):
    hs = torch.randn(sl, 1536, dtype=torch.bfloat16, device=dev)
    w = torch.randn(4608, 1536, dtype=torch.bfloat16, device=dev) * 0.05
    b = torch.randn(4608, dtype=torch.bfloat16, device=dev)
    return hs, w, b


def check(a, b, tol):
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    thresh = tol["max_atol"] + tol["max_rtol"] * b.abs()
    matched = (diff <= thresh).float().mean().item()
    return matched, diff.max().item()


def bench(fn, args, iters=20):
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


allok = True
for sl, tol in zip(seqs, tols):
    hs, w, b = make(sl)
    ref = reference.run(hs, w, b)
    got = kernel.run(hs, w, b)
    line = f"seq={sl:5d} "
    ok = True
    for i, (r, g) in enumerate(zip(ref, got)):
        assert r.shape == g.shape, (r.shape, g.shape)
        m, d = check(g, r, tol)
        if m < tol["required_matched_ratio"]:
            ok = False
        line += f" [{i}] match={m:.5f} maxdiff={d:.5f}"
    tr = bench(reference.run, (hs, w, b), 5)
    tk = bench(kernel.run, (hs, w, b))
    line += f"  ref={tr:8.3f}ms mine={tk:7.3f}ms  x{tr/tk:6.1f}"
    print(("PASS " if ok else "FAIL ") + line, flush=True)
    allok &= ok

print("ALL PASS" if allok else "SOME FAIL")
