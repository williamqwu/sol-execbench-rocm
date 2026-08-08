import sys, torch, time, triton, triton.language as tl
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import tk
dev = 'cuda'
FP8 = tl.float8e4nv
EMAX = tl.constexpr(448.0)
RMAX = tl.constexpr(1.0 / 448.0)


def bench(fn, n=200, w=50):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


# fused: both weights in one launch. grid = (nblocks_total, KB) is ragged;
# instead flatten: program id -> (which tensor, n_blk, k_blk)
@triton.jit
def _wq2(W1, Q1, S1, W2, Q2, S2, NB1, KB1, KB2, s1n, s1s, s2n, s2s,
         NPROG1: tl.constexpr):
    pid = tl.program_id(0)
    if pid < NPROG1:
        nb = pid // KB1
        kb = pid % KB1
        rn = nb * 128 + tl.arange(0, 128)
        rk = kb * 128 + tl.arange(0, 128)
        p = rn[:, None] * s1n + rk[None, :]
        w = tl.load(W1 + p).to(tl.float32)
        sc = tl.maximum(tl.max(tl.abs(w)) * RMAX, 1e-12)
        q = tl.minimum(tl.maximum(w / sc, -EMAX), EMAX)
        tl.store(Q1 + p, q.to(FP8))
        tl.store(S1 + nb * s1s + kb, sc)
    else:
        pid2 = pid - NPROG1
        nb = pid2 // KB2
        kb = pid2 % KB2
        rn = nb * 128 + tl.arange(0, 128)
        rk = kb * 128 + tl.arange(0, 128)
        p = rn[:, None] * s2n + rk[None, :]
        w = tl.load(W2 + p).to(tl.float32)
        sc = tl.maximum(tl.max(tl.abs(w)) * RMAX, 1e-12)
        q = tl.minimum(tl.maximum(w / sc, -EMAX), EMAX)
        tl.store(Q2 + p, q.to(FP8))
        tl.store(S2 + nb * s2s + kb, sc)


# variant: 128 x 256 tile per program (2 k-blocks)
@triton.jit
def _wq_k2(W, Q, S, KB, sn, ss):
    pid = tl.program_id(0)
    nb = pid // (KB // 2)
    kb = (pid % (KB // 2)) * 2
    rn = nb * 128 + tl.arange(0, 128)
    rk = kb * 128 + tl.arange(0, 256)
    p = rn[:, None] * sn + rk[None, :]
    w = tl.load(W + p).to(tl.float32)
    w2 = tl.reshape(w, (128, 2, 128))
    a = tl.max(tl.abs(w2), axis=2)
    a = tl.max(a, axis=0)
    sc = tl.maximum(a * RMAX, 1e-12)
    sce = tl.reshape(tl.broadcast_to(sc[None, :, None], (128, 2, 128)), (128, 256))
    q = tl.minimum(tl.maximum(w / sce, -EMAX), EMAX)
    tl.store(Q + p, q.to(FP8))
    tl.store(S + nb * ss + kb + tl.arange(0, 2), sc)


H, I = 3584, 2048
guw = torch.randn(2 * I, H, device=dev, dtype=torch.bfloat16)
dw = torch.randn(H, I, device=dev, dtype=torch.bfloat16)
q1 = torch.empty_like(guw, dtype=torch.float8_e4m3fn)
s1 = torch.empty((2 * I // 128, H // 128), dtype=torch.float32, device=dev)
q2 = torch.empty_like(dw, dtype=torch.float8_e4m3fn)
s2 = torch.empty((H // 128, I // 128), dtype=torch.float32, device=dev)
NB1, KB1 = 2 * I // 128, H // 128
NB2, KB2 = H // 128, I // 128
NP1 = NB1 * KB1
TOT = NP1 + NB2 * KB2
BYTES = (2 * I * H + H * I) * 3

t = bench(lambda: (tk.quant_weight(guw), tk.quant_weight(dw)))
print(f"separate (alloc incl): {t*1e3:.1f} us")
for nw in [2, 4, 8]:
    t = bench(lambda: (
        tk._quant_w_128x128[(NB1, KB1)](guw, q1, s1, 2 * I, H, guw.stride(0), s1.stride(0), num_warps=nw),
        tk._quant_w_128x128[(NB2, KB2)](dw, q2, s2, H, I, dw.stride(0), s2.stride(0), num_warps=nw)))
    print(f"separate noalloc nw={nw}: {t*1e3:.1f} us ({BYTES/(t/1e3)/1e12:.2f} TB/s)")
for nw in [2, 4, 8]:
    t = bench(lambda: _wq2[(TOT,)](guw, q1, s1, dw, q2, s2, NB1, KB1, KB2,
                                   guw.stride(0), s1.stride(0), dw.stride(0),
                                   s2.stride(0), NPROG1=NP1, num_warps=nw))
    print(f"fused nw={nw}: {t*1e3:.1f} us ({BYTES/(t/1e3)/1e12:.2f} TB/s)")
_wq2[(TOT,)](guw, q1, s1, dw, q2, s2, NB1, KB1, KB2, guw.stride(0), s1.stride(0),
             dw.stride(0), s2.stride(0), NPROG1=NP1, num_warps=4)
r1, rs1 = tk.quant_weight(guw)
r2, rs2 = tk.quant_weight(dw)
print("fused correct:", (q1.view(torch.uint8) == r1.view(torch.uint8)).all().item(),
      (q2.view(torch.uint8) == r2.view(torch.uint8)).all().item(),
      (s1 == rs1).all().item(), (s2 == rs2).all().item())

for nw in [4, 8]:
    t = bench(lambda: (
        _wq_k2[(NB1 * KB1 // 2,)](guw, q1, s1, KB1, guw.stride(0), s1.stride(0), num_warps=nw),
        _wq_k2[(NB2 * KB2 // 2,)](dw, q2, s2, KB2, dw.stride(0), s2.stride(0), num_warps=nw)))
    print(f"k2 nw={nw}: {t*1e3:.1f} us ({BYTES/(t/1e3)/1e12:.2f} TB/s)")
_wq_k2[(NB1 * KB1 // 2,)](guw, q1, s1, KB1, guw.stride(0), s1.stride(0), num_warps=4)
print("k2 correct:", (q1.view(torch.uint8) == r1.view(torch.uint8)).all().item(),
      (s1 == rs1).all().item())
