import sys, torch, triton, triton.language as tl, itertools
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import tk5 as tk
dev = 'cuda'
FP8 = tl.float8e4nv
EMAX = tl.constexpr(448.0)
RMAX = tl.constexpr(1.0 / 448.0)


def bench(fn, n=50, w=20):
    try:
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
    except Exception:
        return 1e9
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in ev)
    return ts[len(ts) // 2]


# weight quant only, R rows of the 128x128 tile per program (R=128 -> 1 prog/tile)
# NR = number of 128x128 tiles handled per program along K
@triton.jit
def wq(W1, Q1, S1, s1n, s1s, IB: tl.constexpr, KB1: tl.constexpr,
       NT: tl.constexpr):
    pid = tl.program_id(0)
    nb1 = pid // (KB1 // NT)
    kg = pid % (KB1 // NT)
    half = nb1 // IB
    j = nb1 % IB
    dst = 2 * j + half
    for t in tl.static_range(NT):
        kb1 = kg * NT + t
        rk1 = kb1 * 128 + tl.arange(0, 128)
        sp = (nb1 * 128 + tl.arange(0, 128))[:, None] * s1n + rk1[None, :]
        dp = (dst * 128 + tl.arange(0, 128))[:, None] * s1n + rk1[None, :]
        w = tl.load(W1 + sp).to(tl.float32)
        c = tl.maximum(tl.max(tl.abs(w)) * RMAX, 1e-12)
        tl.store(Q1 + dp, tl.minimum(tl.maximum(w / c, -EMAX), EMAX).to(FP8))
        tl.store(S1 + dst * s1s + kb1, c)


H, I = 3584, 2048
guw = torch.randn(2 * I, H, device=dev, dtype=torch.bfloat16)
wq_o = torch.empty((2 * I, H), dtype=torch.float8_e4m3fn, device=dev)
wsc = torch.empty((2 * I // 128, H // 128), device=dev)
KB1 = H // 128
BYTES = 2 * I * H * 2 + 2 * I * H + (2 * I // 128) * KB1 * 4
r = []
for nt, nw, ns in itertools.product([1, 2, 4, 7], [1, 2, 4, 8], [1, 2, 3]):
    if KB1 % nt:
        continue
    npg = (2 * I // 128) * (KB1 // nt)
    t = bench(lambda: wq[(npg,)](guw, wq_o, wsc, guw.stride(0), wsc.stride(0),
                                 IB=I // 128, KB1=KB1, NT=nt, num_warps=nw,
                                 num_stages=ns))
    r.append((t, (nt, nw, ns)))
r.sort()
print("gate_up weight quant (current shape = NT=1,nw=4):")
for t, c in r[:6]:
    print(f"   NT,nw,ns={c} {t*1e3:6.1f}us  {BYTES/(t/1e3)/1e12:.2f} TB/s")
