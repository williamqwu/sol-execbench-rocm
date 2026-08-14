import torch, time, sys, json
import triton
import triton.language as tl

DEV = "cuda:0"
H = 3072
TOPK = 8

MS = [131, 256, 512, 1024, 2048, 2164, 3758, 4096, 8192]


def make(M, seed=0):
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    final = torch.randn(M, H, dtype=torch.bfloat16, device=DEV, generator=g)
    src = torch.randn(M * TOPK, H, dtype=torch.bfloat16, device=DEV, generator=g)
    idx = torch.randint(0, M, (M * TOPK,), dtype=torch.long, device=DEV, generator=g)
    return final, src, idx


def ref(final, src, idx):
    out = final.clone()
    out.index_add_(0, idx, src)
    return out


def exact(final, src, idx):
    out = final.float().clone()
    out.index_add_(0, idx, src.float())
    return out.to(torch.bfloat16)


def tol_check(a, b, atol, rtol, ratio=0.99):
    a = a.float(); b = b.float()
    d = (a - b).abs()
    ok = (d <= atol) | (d <= rtol * b.abs())
    return ok.float().mean().item(), d.max().item()


def bench(fn, args, iters=50, warmup=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(*args); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def sol_us(M):
    b = 10 * M * H * 2 + TOPK * M * 8
    return b / 8e12 * 1e6


if __name__ == "__main__":
    import importlib
    mods = sys.argv[1:] or []
    cands = {"ref": ref}
    for m in mods:
        mod = importlib.import_module(m)
        cands[m] = mod.run
    print(f"{'M':>6} {'SOL':>8} " + " ".join(f"{k:>10}" for k in cands))
    for M in MS:
        final, src, idx = make(M)
        r = ref(final, src, idx)
        ex = exact(final, src, idx)
        row = []
        for k, fn in cands.items():
            o = fn(final, src, idx)
            rr, dm = tol_check(o, r, 0.234375, 0.375)
            t = bench(fn, (final, src, idx))
            row.append(f"{t:10.1f}")
            if rr < 0.99:
                row[-1] = f"BAD{rr:.3f}"
        print(f"{M:6d} {sol_us(M):8.1f} " + " ".join(row))
