"""Local correctness + latency check. Not collected; scratch tooling."""
import json, sys, time, importlib
import torch

sys.path.insert(0, ".")
import reference

WL = [json.loads(l) for l in open("workload.jsonl") if l.strip()]


def make_inputs(B, S, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    u = torch.randn((B, 768, S), device="cuda", dtype=torch.float32, generator=g)
    w = torch.randn((768, 1, 3), device="cuda", dtype=torch.float32, generator=g)
    b = torch.randn((768,), device="cuda", dtype=torch.float32, generator=g)
    return u, w, b


def check(ref, got, atol, rtol, ratio):
    ok = True
    worst = []
    for r, g in zip(ref, got):
        assert r.shape == g.shape, (r.shape, g.shape)
        assert r.dtype == g.dtype, (r.dtype, g.dtype)
        d = (r.double() - g.double()).abs()
        thr = atol + rtol * r.double().abs()
        matched = (d <= thr).double().mean().item()
        # normalized error like harness likely reports
        rel = (d / thr).max().item()
        worst.append((matched, rel))
        if matched < ratio:
            ok = False
    return ok, worst


def bench(fn, args, iters=200):
    for _ in range(20):
        fn(*args)
    torch.cuda.synchronize()
    # graph-free timing with events
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    torch.cuda.synchronize()
    st.record()
    for _ in range(iters):
        fn(*args)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000.0  # us


def main():
    import kernel
    importlib.reload(kernel)
    allok = True
    tot_mine = 0.0
    tot_ref = 0.0
    for wl in WL:
        B = wl["axes"]["batch_size"]; S = wl["axes"]["seq_len"]
        tol = wl["tolerance"]
        u, w, b = make_inputs(B, S)
        ref = reference.run(u, w, b)
        got = kernel.run(u, w, b)
        ok, worst = check(ref, got, tol["max_atol"], tol["max_rtol"], tol["required_matched_ratio"])
        tm = bench(kernel.run, (u, w, b))
        tr = bench(reference.run, (u, w, b))
        tot_mine += tm; tot_ref += tr
        bytes_ = 2 * B * 768 * S * 4
        gbs = bytes_ / (tm * 1e-6) / 1e9
        allok &= ok
        print(f"B={B:3d} S={S:5d}  {'PASS' if ok else 'FAIL'}  "
              f"match={min(m for m,_ in worst):.5f} relerr={max(r for _,r in worst):8.3f}  "
              f"mine={tm:8.2f}us ref={tr:8.2f}us  speedup={tr/tm:5.2f}x  {gbs:7.0f} GB/s")
    print(f"TOTAL mine={tot_mine:.1f}us ref={tot_ref:.1f}us  ({tot_ref/tot_mine:.2f}x)  allok={allok}")


if __name__ == "__main__":
    main()
