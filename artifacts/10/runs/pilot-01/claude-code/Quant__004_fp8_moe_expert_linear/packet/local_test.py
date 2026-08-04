import json, sys, time, importlib, enum
import torch

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self): return self.value
    enum.StrEnum = StrEnum

import reference as ref

def load_kernel():
    import kernel
    importlib.reload(kernel)
    return kernel

def check(out, exp, atol, rtol, ratio):
    out = out.float(); exp = exp.float()
    err = (out - exp).abs()
    allowed = atol + rtol * exp.abs()
    ok = err <= allowed
    matched = ok.float().mean().item()
    # worst relative excess
    return matched, err.max().item(), (err - allowed).max().item()

def bench(fn, args, iters=30):
    for _ in range(5): fn(*args)
    torch.cuda.synchronize()
    # use events
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters): fn(*args)
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/iters

def main():
    wls = [json.loads(l) for l in open("workload.jsonl")]
    tokens = sorted({w["axes"]["num_tokens"] for w in wls})
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        tokens = [384, 1024, 3584]
    k = load_kernel()
    tolmap = {w["axes"]["num_tokens"]: w["tolerance"] for w in wls}
    dev = torch.device("cuda:0")
    tot_r = tot_k = 0.0
    for n in tokens:
        torch.manual_seed(n)
        inp = ref.get_inputs({"num_tokens": n}, dev)
        a = (inp["hidden_states"], inp["routing_weight"], inp["gate_up_weight"], inp["down_weight"])
        exp = ref.run(*a)
        got = k.run(*a)
        assert got.shape == exp.shape and got.dtype == exp.dtype, (got.shape, got.dtype, exp.shape, exp.dtype)
        t = tolmap[n]
        matched, maxerr, excess = check(got, exp, t["max_atol"], t["max_rtol"], t["required_matched_ratio"])
        tr = bench(ref.run, a, 10)
        tk = bench(k.run, a, 30)
        tot_r += tr; tot_k += tk
        flag = "PASS" if matched >= t["required_matched_ratio"] else "FAIL"
        print(f"n={n:5d} {flag} matched={matched:.6f} maxerr={maxerr:.5f} excess={excess:+.5f} "
              f"ref={tr:8.3f}ms mine={tk:7.3f}ms  x{tr/tk:6.1f}")
    print(f"TOTAL ref={tot_r:.2f} mine={tot_k:.3f}")

main()
