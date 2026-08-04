import json, sys, time, importlib
import torch

sys.path.insert(0, ".")
import reference

DEV = "cuda:0"


def make_inputs(bs, sl, silu, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    H, E = 768, 1536

    def r(*shape):
        return torch.randn(*shape, generator=g, device=DEV, dtype=torch.float32).to(torch.bfloat16)

    return (r(bs, sl, H), r(bs, sl, E), r(bs, sl, E), r(H, E), r(H), r(bs, sl, E), r(bs, sl, E), silu)


def check(a, b, atol, rtol, ratio, name):
    a = a.float()
    b = b.float()
    ok = (a - b).abs() <= (atol + rtol * b.abs())
    m = ok.float().mean().item()
    err = (a - b).abs().max().item()
    rel = ((a - b).abs() / (b.abs() + 1e-30)).max().item()
    return m >= ratio, m, err, rel, name


def bench(fn, args, iters=30):
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    # use events
    st = torch.cuda.Event(True)
    en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn(*args)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000  # us


def main():
    import kernel
    importlib.reload(kernel)
    wls = [json.loads(l) for l in open("workload.jsonl")]
    only = None
    if len(sys.argv) > 1:
        only = [int(x) for x in sys.argv[1].split(",")]
    tot_ref = tot_mine = 0.0
    allpass = True
    for i, w in enumerate(wls):
        if only is not None and i not in only:
            continue
        bs, sl = w["axes"]["batch_size"], w["axes"]["seq_len"]
        silu = w["inputs"]["use_silu_gate"]["value"]
        tol = w["tolerance"]
        args = make_inputs(bs, sl, silu, seed=i)
        ref = reference.run(*args)
        mine = kernel.run(*args)
        msgs = []
        okall = True
        for r, m, nm in zip(ref, mine, ["gssm", "ggate", "gw", "gb"]):
            if r.shape != m.shape:
                msgs.append(f"{nm}:SHAPE {r.shape}!={m.shape}")
                okall = False
                continue
            if r.dtype != m.dtype:
                msgs.append(f"{nm}:DTYPE {r.dtype}!={m.dtype}")
                okall = False
            ok, ratio, err, rel, _ = check(m, r, tol["max_atol"], tol["max_rtol"], tol["required_matched_ratio"], nm)
            if not ok:
                okall = False
            msgs.append(f"{nm}:{'ok' if ok else 'FAIL'} ratio={ratio:.5f} maxabs={err:.4g}")
        tr = bench(reference.run, args)
        tm = bench(kernel.run, args)
        tot_ref += tr
        tot_mine += tm
        allpass &= okall
        print(f"[{i:2d}] bs={bs:3d} sl={sl:5d} M={bs*sl:6d} silu={int(silu)} "
              f"ref={tr:9.1f}us mine={tm:9.1f}us  x{tr/tm:5.2f}  {'PASS' if okall else 'FAIL'}")
        if not okall:
            print("      " + " | ".join(msgs))
    print(f"TOTAL ref={tot_ref:.1f}us mine={tot_mine:.1f}us speedup={tot_ref/tot_mine:.2f} allpass={allpass}")


main()
