import torch, triton, triton.language as tl, itertools, sys

SHAPES = [
    (1,512,12288),(4,2048,12288),(32,128,12288),(2,211,8192),(1,8192,4096),
    (1,1024,16384),(16,1163,8192),(4,541,8192),(4,449,4096),(64,1024,8192),
    (2,131,4096),(2,293,12288),
]

@triton.jit
def k_row(X, Y, mult, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    base = row * N
    mask = cols < N
    x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) * (1.0 / N)
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) * (1.0 / N)
    thr = mean + tl.sqrt(var) * mult
    tl.store(Y + base + cols, tl.maximum(x - thr, 0.0).to(tl.bfloat16), mask=mask)


# variant: R rows per program, 2D
@triton.jit
def k_rows2d(X, Y, mult, nrows, N: tl.constexpr, BLOCK: tl.constexpr, R: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * R + tl.arange(0, R)
    m = r < nrows
    base = r.to(tl.int64)[:, None] * N + tl.arange(0, BLOCK)[None, :]
    cm = (tl.arange(0, BLOCK) < N)[None, :] & m[:, None]
    x = tl.load(X + base, mask=cm, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) * (1.0 / N)
    d = tl.where(cm, x - mean[:, None], 0.0)
    var = tl.sum(d * d, axis=1) * (1.0 / N)
    thr = mean + tl.sqrt(var) * mult
    tl.store(Y + base, tl.maximum(x - thr[:, None], 0.0).to(tl.bfloat16), mask=cm)


# variant: loop over chunks, 2 passes (reload from cache)
@triton.jit
def k_loop(X, Y, mult, N, NB: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = row * N
    s = tl.zeros([BLOCK], tl.float32)
    s2 = tl.zeros([BLOCK], tl.float32)
    for i in range(NB):
        o = i * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + base + o, mask=o < N, other=0.0).to(tl.float32)
        s += x
        s2 += x * x
    mean = tl.sum(s, axis=0) * (1.0 / N)
    var = tl.sum(s2, axis=0) * (1.0 / N) - mean * mean
    thr = mean + tl.sqrt(tl.maximum(var, 0.0)) * mult
    for i in range(NB):
        o = i * BLOCK + tl.arange(0, BLOCK)
        m = o < N
        x = tl.load(X + base + o, mask=m, other=0.0).to(tl.float32)
        tl.store(Y + base + o, tl.maximum(x - thr, 0.0).to(tl.bfloat16), mask=m)


def bench(fn, n=50):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(n): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/n

mult = -1.2815515

for (B,S,N) in SHAPES:
    x = torch.randn(B*S, N, device='cuda', dtype=torch.bfloat16)
    out = torch.empty_like(x)
    rows = B*S
    gb = x.numel()*2*2/1e9
    best = None
    res = []
    blk = triton.next_power_of_2(N)
    for w in (2,4,8,16):
        try:
            t = bench(lambda: k_row[(rows,)](x,out,mult,N=N,BLOCK=blk,num_warps=w,num_stages=1))
            res.append((t, f"row w{w}"))
        except Exception as e: pass
    for R in (2,4):
        for w in (4,8,16):
            if blk*R//64//w < 1: continue
            try:
                g = (triton.cdiv(rows,R),)
                t = bench(lambda: k_rows2d[g](x,out,mult,rows,N=N,BLOCK=blk,R=R,num_warps=w,num_stages=1))
                res.append((t, f"2d R{R} w{w}"))
            except Exception as e: pass
    for bs in (1024,2048,4096):
        if bs >= blk: continue
        nb = triton.cdiv(N,bs)
        for w in (4,8):
            try:
                t = bench(lambda: k_loop[(rows,)](x,out,mult,N,NB=nb,BLOCK=bs,num_warps=w,num_stages=2))
                res.append((t, f"loop b{bs} w{w}"))
            except Exception as e: pass
    res.sort()
    print(f"{B}x{S}x{N} rows={rows} {gb:.3f}GB : " + " | ".join(f"{n}={t*1000:.1f}us({gb/t*1000:.1f}TB/s)" for t,n in res[:4]))
    sys.stdout.flush()
