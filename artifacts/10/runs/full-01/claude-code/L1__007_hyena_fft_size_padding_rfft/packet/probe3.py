import torch, triton, triton.language as tl, time, json
dev = 'cuda:0'


@triton.jit
def pad_k(X, OUT, S, N, TOT, BLOCK: tl.constexpr):
    # OUT is (rows, N); X is (rows, S). Write x then zeros.
    pid = tl.program_id(0)
    offs = pid*BLOCK + tl.arange(0, BLOCK)
    m = offs < TOT
    row = offs // N
    col = offs % N
    inr = col < S
    v = tl.load(X + row*S + col, mask=m & inr, other=0.0)
    tl.store(OUT + offs, v, mask=m)


@triton.jit
def pad_k2(X, OUT, S, N, ROWS, BLOCK: tl.constexpr):
    # one program per (row, chunk of S); copy chunk. zeros done separately
    pid = tl.program_id(0)
    row = tl.program_id(1)
    offs = pid*BLOCK + tl.arange(0, BLOCK)
    m = offs < S
    v = tl.load(X + row*S + offs, mask=m, other=0.0)
    tl.store(OUT + row*N + offs, v, mask=m)


@triton.jit
def ep(SRC, RE, IM, M, INV, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid*BLOCK + tl.arange(0, BLOCK); m = offs < M
    two = tl.arange(0, 2)
    v = tl.load(SRC + offs[:, None]*2 + two[None, :], mask=m[:, None], other=0.)
    re, im = tl.split(v)
    tl.store(RE + offs, re*INV, mask=m); tl.store(IM + offs, im*INV, mask=m)


def bench(f, iters=25, warm=8):
    for _ in range(warm): f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter(); f(); torch.cuda.synchronize()
        ts.append(time.perf_counter()-t0)
    ts.sort(); return ts[len(ts)//2]*1e3


rows_w = [json.loads(l) for l in open('workload.jsonl')]
CASES = [(w['axes']['batch_size'], w['axes']['seqlen']) for w in rows_w]

print("=== PAD strategies (ms) ===")
print(f"{'B':>3}{'S':>7} {'rfft(n=)':>9} {'t.pad':>7} {'zeros+cp':>9} {'triPad':>7} {'triPad2':>8} {'rfftPad':>8} | {'best pad+rfft':>13}")
for (B, S) in CASES:
    C = 256; N = 2*S; R = B*C
    x = torch.randn(B, C, S, device=dev, dtype=torch.float32)
    x2 = x.reshape(R, S)
    t_rfft_n = bench(lambda: torch.fft.rfft(x, n=N))
    t_tpad = bench(lambda: torch.nn.functional.pad(x, (0, S)))

    def zc():
        b = torch.zeros(R, N, device=dev, dtype=torch.float32)
        b[:, :S] = x2
        return b
    t_zc = bench(zc)

    buf = torch.empty(R, N, device=dev, dtype=torch.float32)
    TOT = R*N
    def tp():
        pad_k[(triton.cdiv(TOT, 1024),)](x2, buf, S, N, TOT, BLOCK=1024)
    t_tp = bench(tp)

    def tp2():
        buf[:, S:].zero_()
        pad_k2[(triton.cdiv(S, 1024), R)](x2, buf, S, N, R, BLOCK=1024)
    t_tp2 = bench(tp2)

    padded = torch.nn.functional.pad(x, (0, S))
    t_rp = bench(lambda: torch.fft.rfft(padded, n=N))
    bestpad = min(t_tpad, t_zc, t_tp, t_tp2)
    print(f"{B:>3}{S:>7} {t_rfft_n:9.3f} {t_tpad:7.3f} {t_zc:9.3f} {t_tp:7.3f} {t_tp2:8.3f} {t_rp:8.3f} | {bestpad+t_rp:13.3f}")
    del x, x2, buf, padded
    torch.cuda.empty_cache()
