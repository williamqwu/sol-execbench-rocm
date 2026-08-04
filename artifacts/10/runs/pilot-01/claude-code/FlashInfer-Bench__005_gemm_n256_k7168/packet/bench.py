"""Local scratch benchmark harness (not part of the solution)."""
import json
import os
import sys
import torch

DEV = "cuda:0"
N, K = 256, 7168
MS = [1, 32, 80, 901, 16, 15, 14, 4, 14104, 11948, 63, 58, 57, 56, 55, 54, 53]


def ref(A, B):
    return torch.matmul(A, B.T)


def make(M, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    A = torch.randn(M, K, generator=g, device=DEV, dtype=torch.float16)
    B = torch.randn(N, K, generator=g, device=DEV, dtype=torch.float16)
    return A, B


def time_fn(fn, args, iters=50, warmup=20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    # use cuda events, take median of repeated batches
    ts = []
    for _ in range(7):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            fn(*args)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / iters * 1000.0)  # us
    ts.sort()
    return ts[len(ts) // 2]


def check(out, exp, atol, rtol, ratio=0.99):
    out = out.float()
    exp = exp.float()
    diff = (out - exp).abs()
    tol = atol + rtol * exp.abs()
    ok = (diff <= tol) | (diff <= atol)
    matched = ok.float().mean().item()
    return matched, diff.max().item(), (diff / (exp.abs() + 1e-9)).max().item()


TOL = {}
with open(os.path.join(os.path.dirname(__file__), "workload.jsonl")) as f:
    for line in f:
        w = json.loads(line)
        TOL[w["axes"]["M"]] = w["tolerance"]


def run_suite(fn, name, ms=None, verify=True, iters=50):
    print(f"=== {name} ===")
    tot = 0.0
    for M in (ms or MS):
        A, B = make(M)
        exp = ref(A, B)
        try:
            out = fn(A, B)
        except Exception as ex:
            print(f"M={M:6d}  ERROR {type(ex).__name__}: {ex}")
            continue
        t = TOL[M]
        matched, mad, mrd = check(out, exp, t["max_atol"], t["max_rtol"])
        status = "PASS" if matched >= t["required_matched_ratio"] else "FAIL"
        us = time_fn(fn, (A, B), iters=iters)
        tot += us
        print(
            f"M={M:6d}  {status}  matched={matched:.5f} maxabs={mad:.5f}"
            f" (atol {t['max_atol']:.4f})  {us:9.2f} us"
        )
    print(f"total: {tot:.2f} us")
    return tot


if __name__ == "__main__":
    run_suite(ref, "torch.matmul baseline")
