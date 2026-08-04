import json, sys, importlib, time
import torch

sys.path.insert(0, ".")
import reference

MOD = sys.argv[1] if len(sys.argv) > 1 else "kernel"
kernel = importlib.import_module(MOD)

CONST = {"d_model": 1280, "num_heads": 20, "head_dim": 64}
NAMES = ["grad_hidden_states", "grad_q_weight", "grad_q_bias", "grad_k_weight",
         "grad_k_bias", "grad_v_weight", "grad_v_bias", "grad_out_weight", "grad_out_bias"]

dev = torch.device("cuda:0")
wls = [json.loads(l) for l in open("workload.jsonl")]

ORDER = ["grad_output", "hidden_states", "query_states", "key_states", "value_states",
         "cu_seqlens", "q_weight", "k_weight", "v_weight", "out_weight"]


def check(ref, got, atol, rtol, ratio):
    ok = True
    rows = []
    for n, r, g in zip(NAMES, ref, got):
        assert r.shape == g.shape, (n, r.shape, g.shape)
        assert r.dtype == g.dtype, (n, r.dtype, g.dtype)
        rf = r.float()
        gf = g.float()
        diff = (rf - gf).abs()
        tol = atol + rtol * rf.abs()
        matched = (diff <= tol).float().mean().item()
        maxerr = diff.max().item()
        allowed = tol.flatten()[diff.flatten().argmax()].item()
        good = matched >= ratio
        ok = ok and good
        rows.append((n, matched, maxerr, allowed, good))
    return ok, rows


def bench(fn, args, iters=20):
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


only = None
if len(sys.argv) > 2:
    only = int(sys.argv[2])

allok = True
for i, w in enumerate(wls):
    if only is not None and i != only:
        continue
    ax = dict(CONST)
    ax.update(w["axes"])
    torch.manual_seed(1234 + i)
    inp = reference.get_inputs(ax, dev)
    args = [inp[k] for k in ORDER]
    ref = reference.run(*args)
    got = kernel.run(*args)
    tolc = w["tolerance"]
    ok, rows = check(ref, got, tolc["max_atol"], tolc["max_rtol"], tolc["required_matched_ratio"])
    allok = allok and ok
    tag = "PASS" if ok else "FAIL"
    print(f"[{i:2d}] N={ax['total_seq_len']:5d} nc={ax['num_chunks']:3d} {tag}")
    for n, m, e, a, g in rows:
        if not g or "-v" in sys.argv:
            print(f"      {'ok ' if g else 'BAD'} {n:20s} matched={m:.5f} maxerr={e:.4g} allowed@max={a:.4g}")
    if "-t" in sys.argv:
        tk = bench(kernel.run, args)
        tr = bench(reference.run, args)
        print(f"      time kernel={tk*1e3:8.1f}us ref={tr*1e3:8.1f}us  speedup={tr/tk:.2f}x")

print("ALL PASS" if allok else "SOME FAILED")
