import torch, triton
import triton.language as tl
from variants import make_run, bench_ev, mk, DEV, HID

# ---- achievable bandwidth ceiling probes -------------------------------------
@triton.jit
def _copy(SRC, DST, n, BS: tl.constexpr):
    pid = tl.program_id(0)
    for off in range(pid * BS, n, tl.num_programs(0) * BS):
        i = off + tl.arange(0, BS)
        m = i < n
        tl.store(DST + i, tl.load(SRC + i, mask=m, other=0.0), mask=m)


def ceiling():
    nbytes = 13.4e9
    n = int(nbytes / 8)  # fp32 read+write = 8 B/elt
    a = torch.empty(n, device=DEV, dtype=torch.float32)
    b = torch.empty(n, device=DEV, dtype=torch.float32)
    for ng in (512, 1024, 2048):
        t = bench_ev(lambda: _copy[(ng,)](a, b, n, BS=4096, num_warps=8), (), iters=20, warmup=5)
        print(f"  copy grid={ng:5d}: {t*1000:8.1f}us  {n*8/(t*1e-3)/1e9:7.0f} GB/s")


MS = [128, 422, 1249, 1546, 3412, 8192, 15952, 34784, 53024, 110144, 131072, 524288]

if __name__ == "__main__":
    print("=== pure copy bandwidth ceiling (8 B/elt) ===")
    ceiling()

    print("\n=== mode=atomic: nprog x num_warps ===")
    NPS = [64, 128, 192, 256, 384, 512, 768, 1024]
    print(f"{'M':>8} | " + " | ".join(f"{p:>5}" for p in NPS))
    best = {}
    for M in MS:
        args = mk(M)
        row = f"{M:8d} | "
        cells = []
        bt, bcfg = 1e9, None
        for p in NPS:
            sub = []
            for nw in (4, 8):
                f = make_run(lambda m, k=p: min(m, k), 1, nw)
                t = bench_ev(f, args)
                sub.append(t)
                if t < bt:
                    bt, bcfg = t, (p, nw)
            cells.append(f"{min(sub)*1000:5.0f}")
        best[M] = (bt, bcfg)
        print(row + " | ".join(cells) + f"   best={bcfg} {bt*1000:.0f}us")
        del args
        torch.cuda.empty_cache()

    print("\n=== best per M ===")
    for M in MS:
        t, cfg = best[M]
        gb = M * HID * 10 / (t * 1e-3) / 1e9
        print(f"M={M:7d} nprog={cfg[0]:5d} warps={cfg[1]} {t*1000:8.1f}us {gb:7.0f} GB/s")
