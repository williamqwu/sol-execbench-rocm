"""Local correctness + timing harness. Not part of the solution."""
import json, sys, time, importlib
import torch

sys.path.insert(0, ".")
import reference

DEV = "cuda:0"
H = 2560


def make_inputs(B, S, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    go = torch.randn(B, S, H, device=DEV, dtype=torch.bfloat16, generator=g)
    x = torch.randn(B, S, H, device=DEV, dtype=torch.float32, generator=g)
    nrm = torch.randn(B, S, H, device=DEV, dtype=torch.float32, generator=g)
    rstd = torch.randn(B, S, 1, device=DEV, dtype=torch.float32, generator=g)
    w = torch.randn(H, device=DEV, dtype=torch.float32, generator=g)
    return go, x, nrm, rstd, w


def check(ref, got, atol, rtol, ratio, label):
    ref = ref.float()
    got = got.float()
    if ref.shape != got.shape:
        return False, f"{label}: shape {got.shape} != {ref.shape}"
    diff = (ref - got).abs()
    allowed = atol + rtol * ref.abs()
    ok = diff <= allowed
    frac = ok.float().mean().item()
    worst = (diff - allowed).max().item()
    return frac >= ratio, f"{label}: matched={frac:.6f} worst_excess={worst:.3e} maxdiff={diff.max().item():.3e}"


def bench(fn, args, iters=30):
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    # use cuda events
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    st.record()
    for _ in range(iters):
        fn(*args)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def main():
    modname = sys.argv[1] if len(sys.argv) > 1 else "kernel"
    mod = importlib.import_module(modname)
    only = None
    if len(sys.argv) > 2:
        only = int(sys.argv[2])

    workloads = [json.loads(l) for l in open("workload.jsonl")]
    tot_ref = 0.0
    tot_got = 0.0
    allok = True
    for i, wl in enumerate(workloads):
        if only is not None and i != only:
            continue
        B = wl["axes"]["batch_size"]
        S = wl["axes"]["seq_len"]
        tol = wl["tolerance"]
        args = make_inputs(B, S, seed=i)
        ref_out = reference.run(*args)
        got_out = mod.run(*args)
        msgs = []
        okall = True
        for name, r, gt in zip(["gh", "gr", "gw"], ref_out, got_out):
            ok, m = check(r, gt, tol["max_atol"], tol["max_rtol"], tol["required_matched_ratio"], name)
            okall &= ok
            msgs.append(m)
        t_ref = bench(reference.run, args)
        t_got = bench(mod.run, args)
        tot_ref += t_ref
        tot_got += t_got
        n = B * S * H
        gb = n * 10 / 1e9  # 6 bytes read + 4 bytes written
        bw = gb / (t_got / 1e3) / 1e3  # TB/s
        allok &= okall
        print(f"[{i:2d}] B={B:3d} S={S:5d} {'PASS' if okall else 'FAIL'} "
              f"ref={t_ref*1000:9.1f}us mine={t_got*1000:9.1f}us "
              f"speedup={t_ref/t_got:6.2f}x bw={bw:6.2f}TB/s")
        if not okall:
            for m in msgs:
                print("      ", m)
    print(f"TOTAL ref={tot_ref*1000:.1f}us mine={tot_got*1000:.1f}us speedup={tot_ref/tot_got:.2f}x  allok={allok}")


main()
