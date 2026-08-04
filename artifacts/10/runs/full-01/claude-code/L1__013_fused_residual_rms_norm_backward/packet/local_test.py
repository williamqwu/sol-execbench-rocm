import json, sys, time, importlib
import torch

sys.path.insert(0, ".")
import reference

MOD = sys.argv[1] if len(sys.argv) > 1 else "kernel"
kernel = importlib.import_module(MOD)

DEV = "cuda:0"
HID = 2560


def make_inputs(b, s, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    go = torch.randn(b, s, HID, generator=g, device=DEV, dtype=torch.bfloat16)
    x = torch.randn(b, s, HID, generator=g, device=DEV, dtype=torch.float32)
    nm = torch.randn(b, s, HID, generator=g, device=DEV, dtype=torch.float32)
    rs = torch.randn(b, s, 1, generator=g, device=DEV, dtype=torch.float32)
    w = torch.randn(HID, generator=g, device=DEV, dtype=torch.float32)
    return go, x, nm, rs, w


def compare(name, out, ref, atol, rtol, ratio_req):
    out = out.float()
    ref = ref.float()
    diff = (out - ref).abs()
    tol = atol + rtol * ref.abs()
    ok = diff <= tol
    matched = ok.float().mean().item()
    # worst violation
    viol = (diff - tol)
    worst = viol.max().item()
    maxerr = diff.max().item()
    passed = matched >= ratio_req
    return passed, f"{name}: matched={matched:.6f} (req {ratio_req}) maxabs={maxerr:.5g} worst_excess={worst:.3g}"


def bench(fn, args, iters=30):
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    # graph-free timing
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


wls = [json.loads(l) for l in open("workload.jsonl")]

allpass = True
tot_mine = 0.0
tot_ref = 0.0
for i, wl in enumerate(wls):
    b = wl["axes"]["batch_size"]
    s = wl["axes"]["seq_len"]
    tol = wl["tolerance"]
    args = make_inputs(b, s, seed=i)
    ref_out = reference.run(*args)
    my_out = kernel.run(*args)
    names = ["grad_hidden_states", "grad_residual", "grad_weight"]
    msgs = []
    wlpass = True
    for n, o, r in zip(names, my_out, ref_out):
        if o.shape != r.shape:
            wlpass = False
            msgs.append(f"{n}: SHAPE {tuple(o.shape)} != {tuple(r.shape)}")
            continue
        if o.dtype != r.dtype:
            wlpass = False
            msgs.append(f"{n}: DTYPE {o.dtype} != {r.dtype}")
            continue
        p, m = compare(n, o, r, tol["max_atol"], tol["max_rtol"], tol["required_matched_ratio"])
        wlpass &= p
        msgs.append(m)
    if my_out[0].data_ptr() == my_out[1].data_ptr():
        msgs.append("WARNING: gh and gr alias")
    tmine = bench(kernel.run, args)
    tref = bench(reference.run, args)
    tot_mine += tmine
    tot_ref += tref
    elts = b * s * HID
    gbps = elts * 10 / (tmine * 1e-3) / 1e9
    allpass &= wlpass
    print(f"[{'PASS' if wlpass else 'FAIL'}] b={b:3d} s={s:5d} M={b*s:7d}  mine={tmine:8.4f}ms ref={tref:8.4f}ms  x{tref/tmine:6.2f}  {gbps:7.1f} GB/s(10B/elt)")
    for m in msgs:
        print("        ", m)

print()
print("ALL PASS" if allpass else "SOME FAILED")
print(f"total mine={tot_mine:.3f}ms ref={tot_ref:.3f}ms  speedup={tot_ref/tot_mine:.2f}x")
