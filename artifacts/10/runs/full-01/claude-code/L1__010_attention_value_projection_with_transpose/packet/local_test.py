import json, sys, importlib, time
import torch

sys.path.insert(0, ".")
import reference

import kernel as K
importlib.reload(K)

DEV = "cuda:0"
ws = [json.loads(l) for l in open("workload.jsonl")]
shapes = sorted(set((w["axes"]["batch_size"], w["axes"]["seq_len"]) for w in ws),
                key=lambda t: t[0] * t[1])
tol = {(w["axes"]["batch_size"], w["axes"]["seq_len"]): w["tolerance"] for w in ws}


def bench(fn, *a, iters=50):
    for _ in range(10):
        fn(*a)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn(*a)
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def check(ref, got, t):
    ref32 = ref.float()
    got32 = got.float()
    err = (ref32 - got32).abs()
    thr = t["max_atol"] + t["max_rtol"] * ref32.abs()
    ok = (err <= thr)
    ratio = ok.float().mean().item()
    return ratio, err.max().item(), (err / (t["max_atol"] + t["max_rtol"] * ref32.abs())).max().item()


torch.manual_seed(0)
w = torch.randn(1024, 5120, device=DEV, dtype=torch.bfloat16)

tot_ref = tot_got = 0.0
allok = True
for (b, s) in shapes:
    h = torch.randn(b, s, 5120, device=DEV, dtype=torch.bfloat16)
    ref = reference.run(h, w)
    got = K.run(h, w)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert got.dtype == ref.dtype
    assert got.is_contiguous()
    r, e, rel = check(ref, got, tol[(b, s)])
    t_ref = bench(reference.run, h, w)
    t_got = bench(K.run, h, w)
    tot_ref += t_ref
    tot_got += t_got
    st = "OK " if r >= 0.99 else "FAIL"
    if r < 0.99:
        allok = False
    print(f"{st} B={b:3d} S={s:5d} M={b*s:6d} match={r:.5f} maxerr={e:.5f} relmax={rel:.2f} "
          f"ref={t_ref*1000:8.1f}us mine={t_got*1000:8.1f}us  x{t_ref/t_got:5.2f}")
    del h, ref, got
    torch.cuda.empty_cache()

print(f"\ntotal ref={tot_ref*1000:.1f}us mine={tot_got*1000:.1f}us  x{tot_ref/tot_got:.2f}  allok={allok}")
