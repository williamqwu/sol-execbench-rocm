import torch, triton, triton.language as tl, time, json
dev = 'cuda:0'


@triton.jit
def ep(SRC, RE, IM, M, INV, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid*BLOCK + tl.arange(0, BLOCK); m = offs < M
    two = tl.arange(0, 2)
    v = tl.load(SRC + offs[:, None]*2 + two[None, :], mask=m[:, None], other=0.)
    re, im = tl.split(v)
    tl.store(RE + offs, re*INV, mask=m); tl.store(IM + offs, im*INV, mask=m)


def bench(f, iters=25, warm=10):
    for _ in range(warm): f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter(); f(); torch.cuda.synchronize()
        ts.append(time.perf_counter()-t0)
    ts.sort(); return ts[len(ts)//2]*1e3


def full_simple(x, S, N, R):
    z = torch.fft.rfft(x, n=N)
    flat = z.view(torch.float32).reshape(-1)
    M = flat.numel()//2
    out = torch.empty((2, M), device=dev, dtype=torch.float32)
    ep[(triton.cdiv(M, 1024),)](flat, out[0], out[1], M, 1.0/N, BLOCK=1024, num_warps=4)
    return out


def full_chunked(x2, S, N, R, K, streams):
    F = S+1
    out = torch.empty((2, R, F), device=dev, dtype=torch.float32)
    cs = (R + K - 1)//K
    main = torch.cuda.current_stream()
    evs = []
    for i in range(K):
        a = i*cs; b = min(R, a+cs)
        if a >= b: break
        st = streams[i % len(streams)]
        st.wait_stream(main)
        with torch.cuda.stream(st):
            z = torch.fft.rfft(x2[a:b], n=N)
            flat = z.view(torch.float32).reshape(-1)
            M = flat.numel()//2
            ep[(triton.cdiv(M, 1024),)](flat, out[0, a:b].reshape(-1), out[1, a:b].reshape(-1),
                                        M, 1.0/N, BLOCK=1024, num_warps=4)
        evs.append(st.record_event())
    for e in evs: main.wait_event(e)
    return out


rows_w = [json.loads(l) for l in open('workload.jsonl')]
CASES = sorted({(w['axes']['batch_size'], w['axes']['seqlen']) for w in rows_w})
streams = [torch.cuda.Stream() for _ in range(4)]

print(f"{'B':>3}{'S':>7} {'simple':>8} {'k2':>8} {'k4':>8} {'k8':>8} | exact")
for (B, S) in CASES:
    C = 256; N = 2*S; R = B*C
    x = torch.randn(B, C, S, device=dev, dtype=torch.float32)
    x2 = x.reshape(R, S)
    g = full_simple(x, S, N, R).reshape(2, R, S+1).clone()
    t0 = bench(lambda: full_simple(x, S, N, R))
    ts = {}
    exact = True
    for K in (2, 4, 8):
        r = full_chunked(x2, S, N, R, K, streams)
        torch.cuda.synchronize()
        if not torch.equal(r, g): exact = False
        ts[K] = bench(lambda K=K: full_chunked(x2, S, N, R, K, streams))
    print(f"{B:>3}{S:>7} {t0:8.3f} {ts[2]:8.3f} {ts[4]:8.3f} {ts[8]:8.3f} | {exact}")
    del x, x2, g
    torch.cuda.empty_cache()
