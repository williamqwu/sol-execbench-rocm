import json, sys, time, importlib
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
import reference

DEV = "cuda:0"
H = 2048


def make_inputs(B, S, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    def rnd(*shape):
        return torch.randn(*shape, generator=g, device=DEV, dtype=torch.bfloat16)
    x = rnd(B, S, H)
    w1 = rnd(3 * H, H)
    b1 = rnd(3 * H)
    cw = rnd(H, 1, 4)
    cb = rnd(H)
    w2 = rnd(H, H)
    b2 = rnd(H)
    return (x, w1, b1, cw, cb, w2, b2)


def check(ref, got, atol, rtol, ratio):
    ref = ref.float(); got = got.float()
    diff = (ref - got).abs()
    thr = atol + rtol * ref.abs()
    ok = diff <= thr
    matched = ok.float().mean().item()
    # relative error metric similar to harness
    return matched, diff.max().item(), (diff / (thr + 1e-30)).max().item()


def bench(fn, n=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        st = torch.cuda.Event(True); en = torch.cuda.Event(True)
        st.record()
        for _ in range(n):
            fn()
        en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) / n * 1000)
    return min(ts)


def main():
    import kernel
    importlib.reload(kernel)
    wls = [json.loads(l) for l in open("workload.jsonl") if l.strip()]
    tot_ref = 0.0; tot_got = 0.0
    allpass = True
    for i, w in enumerate(wls):
        B = w["axes"]["batch_size"]; S = w["axes"]["seq_len"]
        tol = w["tolerance"]
        inp = make_inputs(B, S, seed=i)
        ref = reference.run(*inp)
        got = kernel.run(*inp)
        assert got.shape == ref.shape, (got.shape, ref.shape)
        assert got.dtype == ref.dtype, (got.dtype, ref.dtype)
        m, mx, rel = check(ref, got, tol["max_atol"], tol["max_rtol"], tol["required_matched_ratio"])
        ok = m >= tol["required_matched_ratio"]
        allpass &= ok
        tr = bench(lambda: reference.run(*inp))
        tg = bench(lambda: kernel.run(*inp))
        tot_ref += tr; tot_got += tg
        print(f"[{'PASS' if ok else 'FAIL'}] B={B:3d} S={S:5d} M={B*S:6d} "
              f"matched={m:.5f} maxdiff={mx:.4f} relworst={rel:6.2f} "
              f"ref={tr:8.1f}us  got={tg:8.1f}us  speedup={tr/tg:5.2f}x")
    print(f"TOTAL ref={tot_ref:.1f}us got={tot_got:.1f}us  speedup={tot_ref/tot_got:.3f}x  allpass={allpass}")


if __name__ == "__main__":
    main()
