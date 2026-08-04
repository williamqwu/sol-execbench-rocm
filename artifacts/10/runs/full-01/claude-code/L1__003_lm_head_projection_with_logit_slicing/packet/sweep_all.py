"""Master sweep: for each workload M, try every impl x pad target, report best."""
import torch, sys, os
import aiter
import tk, tk2

H, V = 2048, 102400
dev = torch.device("cuda:0")
torch.manual_seed(0)
w = (torch.randn(V, H, dtype=torch.bfloat16, device=dev) * (1.0 / H**0.5))
wt = w.t()
KN256 = "_ZN5aiter24bf16gemm_bf16_tn_256x256E"

def timeit(fn, iters=8, warmup=3, reps=3):
    try:
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
    except Exception:
        return None
    ts = []
    for _ in range(reps):
        st = torch.cuda.Event(True); en = torch.cuda.Event(True)
        st.record()
        for _ in range(iters): fn()
        en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) / iters * 1e3)
    return min(ts)

TRI_CFGS = [
    dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, num_warps=8, num_stages=3),
    dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2),
    dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, num_warps=8, num_stages=2),
    dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=3),
    dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=32, num_warps=8, num_stages=3),
    dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=3),
    dict(BLOCK_M=256, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3),
    dict(BLOCK_M=512, BLOCK_N=64, BLOCK_K=64, num_warps=8, num_stages=2),
    dict(BLOCK_M=512, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=2),
]

def pad_targets(M):
    """Candidate padded row counts >= M, at most 1.6x M."""
    cands = {M}
    for base in (64, 128, 256, 384, 512, 768, 1024):
        p = ((M + base - 1) // base) * base
        if p <= int(M * 1.6) + 8:
            cands.add(p)
    return sorted(cands)

WL = [int(x) for x in sys.argv[1].split(",")]
MAXPAD = max(max(pad_targets(m)) for m in WL)
Abuf = torch.randn(MAXPAD, H, dtype=torch.bfloat16, device=dev)
Obuf = torch.empty(MAXPAD, V, dtype=torch.bfloat16, device=dev)

for M in WL:
    x = Abuf[:M]
    ref = torch.mm(x, wt)
    res = []
    for P in pad_targets(M):
        xp = Abuf[:P]
        op = Obuf[:P]
        tag = "" if P == M else f"+pad{P}"
        # torch
        t = timeit(lambda: torch.mm(xp, wt, out=op))
        if t: res.append((t, f"torch{tag}"))
        # aiter default + explicit 256x256
        for nm, kn in (("aiter", None), ("aiter256", KN256)):
            try:
                Obuf[:P].zero_()
                aiter.gemm_a16w16_asm(xp, w, op, None, None, kn, False)
                torch.cuda.synchronize()
                if (op[:M].float() - ref.float()).abs().max().item() > 0.02:
                    continue
                t = timeit(lambda: aiter.gemm_a16w16_asm(xp, w, op, None, None, kn, False))
                if t: res.append((t, f"{nm}{tag}"))
            except Exception:
                pass
        # triton
        for ci, cfg in enumerate(TRI_CFGS):
            for kind, fn in (("g", tk.gemm), ("p", tk2.persist)):
                try:
                    c = dict(cfg, GROUP_M=8) if kind == "g" else cfg
                    fn(xp, wt, c, op)
                    torch.cuda.synchronize()
                    if (op[:M].float() - ref.float()).abs().max().item() > 0.02:
                        continue
                    t = timeit(lambda: fn(xp, wt, c, op))
                    if t:
                        res.append((t, f"tri{kind}{ci}[{cfg['BLOCK_M']}x{cfg['BLOCK_N']}x"
                                       f"{cfg['BLOCK_K']}w{cfg['num_warps']}s{cfg['num_stages']}]{tag}"))
                except Exception:
                    pass
    res.sort(key=lambda r: r[0])
    tor = min([r for r in res if r[1] == "torch"], default=(0, ""))[0]
    print(f"### M={M}  torch={tor:.1f}us")
    for t, nm in res[:7]:
        print(f"   {t:8.1f}us {2*M*H*V/t*1e-6:7.1f}TF  {nm}")
    sys.stdout.flush()
    del ref; torch.cuda.empty_cache()
