import torch, triton, triton.language as tl, sys

def bench(fn, n=50):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(n): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en)/n

@triton.jit
def k_copy(X, Y, n, BLOCK: tl.constexpr):
    o = tl.program_id(0).to(tl.int64)*BLOCK + tl.arange(0,BLOCK)
    tl.store(Y+o, tl.load(X+o, mask=o<n), mask=o<n)

@triton.jit
def k_row(X, Y, mult, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    base = row * N
    x = tl.load(X + base + cols).to(tl.float32)
    mean = tl.sum(x, axis=0) * (1.0 / N)
    d = x - mean
    var = tl.sum(d * d, axis=0) * (1.0 / N)
    thr = mean + tl.sqrt(var) * mult
    tl.store(Y + base + cols, tl.maximum(x - thr, 0.0).to(tl.bfloat16))

# keep x in bf16 regs, convert twice
@triton.jit
def k_row_bf(X, Y, mult, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    base = row * N
    xb = tl.load(X + base + cols)
    x = xb.to(tl.float32)
    mean = tl.sum(x, axis=0) * (1.0 / N)
    d = x - mean
    var = tl.sum(d * d, axis=0) * (1.0 / N)
    thr = mean + tl.sqrt(var) * mult
    tl.store(Y + base + cols, tl.maximum(xb.to(tl.float32) - thr, 0.0).to(tl.bfloat16))

# sum/sumsq single pass (no two-term dependency)
@triton.jit
def k_row_ss(X, Y, mult, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    base = row * N
    x = tl.load(X + base + cols).to(tl.float32)
    s = tl.sum(x, axis=0)
    s2 = tl.sum(x*x, axis=0)
    mean = s * (1.0/N)
    var = s2*(1.0/N) - mean*mean
    thr = mean + tl.sqrt(tl.maximum(var,0.0)) * mult
    tl.store(Y + base + cols, tl.maximum(x - thr, 0.0).to(tl.bfloat16))

# persistent grid-stride over rows
@triton.jit
def k_pers(X, Y, mult, nrows, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    npr = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    for r in range(pid, nrows, npr):
        base = r.to(tl.int64) * N
        x = tl.load(X + base + cols).to(tl.float32)
        mean = tl.sum(x, axis=0) * (1.0 / N)
        d = x - mean
        var = tl.sum(d * d, axis=0) * (1.0 / N)
        thr = mean + tl.sqrt(var) * mult
        tl.store(Y + base + cols, tl.maximum(x - thr, 0.0).to(tl.bfloat16))

# split row across P programs? -> two-pass loop with cg cache hint
@triton.jit
def k_loop(X, Y, mult, N, NB: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = row * N
    s = tl.zeros([BLOCK], tl.float32); s2 = tl.zeros([BLOCK], tl.float32)
    for i in range(NB):
        o = i*BLOCK + tl.arange(0,BLOCK)
        x = tl.load(X+base+o).to(tl.float32)
        s += x; s2 += x*x
    mean = tl.sum(s,axis=0)*(1.0/N)
    var = tl.sum(s2,axis=0)*(1.0/N) - mean*mean
    thr = mean + tl.sqrt(tl.maximum(var,0.0))*mult
    for i in range(NB):
        o = i*BLOCK + tl.arange(0,BLOCK)
        x = tl.load(X+base+o).to(tl.float32)
        tl.store(Y+base+o, tl.maximum(x-thr,0.0).to(tl.bfloat16))

mult = -1.2815515
for (rows, N) in [(65536,8192),(18608,8192),(8192,12288),(8192,4096),(1024,16384)]:
    x = torch.randn(rows, N, device='cuda', dtype=torch.bfloat16)
    out = torch.empty_like(x)
    byt = x.numel()*2*2
    res=[]
    nel = x.numel()
    t = bench(lambda: k_copy[(triton.cdiv(nel,8192),)](x,out,nel,BLOCK=8192,num_warps=8))
    res.append((t,"COPY"))
    blk = N
    for w in (8,16,32):
        for f,nm in ((k_row,"row"),(k_row_bf,"rowbf"),(k_row_ss,"rowss")):
            try:
                t = bench(lambda: f[(rows,)](x,out,mult,N=N,BLOCK=blk,num_warps=w,num_stages=1))
                res.append((t,f"{nm} w{w}"))
            except Exception: pass
    for np_ in (256,512,1024,2048):
        for w in (8,16):
            try:
                t = bench(lambda: k_pers[(np_,)](x,out,mult,rows,N=N,BLOCK=blk,num_warps=w,num_stages=1))
                res.append((t,f"pers{np_} w{w}"))
            except Exception: pass
    for bs in (2048,4096):
        for w in (4,8,16):
            try:
                t = bench(lambda: k_loop[(rows,)](x,out,mult,N,NB=N//bs,BLOCK=bs,num_warps=w,num_stages=2))
                res.append((t,f"loop{bs} w{w}"))
            except Exception: pass
    res.sort()
    print(f"rows={rows} N={N}: " + " | ".join(f"{n}={t*1000:.1f}us({byt/t/1e9:.2f}TB/s)" for t,n in res[:6]))
    sys.stdout.flush()
