import torch, triton, triton.language as tl, time, sys, importlib

SHAPES = [(4,512),(2,691),(1,4096),(16,373),(64,541),(8,512),(8,256),(2,512),
          (1,512),(16,128),(4,1024),(16,512),(1,2048),(8,853),(8,128),(1,1024)]

@triton.jit
def _empty(x):
    pass

def bench(fn, *a, iters=2000, warmup=500):
    for _ in range(warmup): fn(*a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1e3

def bench_gpu(fn, *a, iters=2000, warmup=500):
    for _ in range(warmup): fn(*a)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn(*a)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters

def ref(freqs, sc):
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos()*sc).to(torch.bfloat16), (emb.sin()*sc).to(torch.bfloat16)

if __name__ == "__main__":
    mod = importlib.import_module(sys.argv[1] if len(sys.argv)>1 else "kernel")
    run = mod.run
    x = torch.randn(4,512,64, device="cuda")
    # empty-kernel launch floor
    for _ in range(100): _empty[(1024,)](x)
    torch.cuda.synchronize()
    print(f"empty-kernel launch floor: wall={bench(lambda: _empty[(1024,)](x)):.4f}ms")
    print(f"torch.empty x2 floor: wall={bench(lambda: (torch.empty((4,512,128),dtype=torch.bfloat16,device='cuda'),torch.empty((4,512,128),dtype=torch.bfloat16,device='cuda'))):.4f}ms")
    print()
    print(f"{'shape':>14} {'rows':>7} {'MB':>7} {'wall(ms)':>9} {'gpu(ms)':>9} {'ref(ms)':>9} {'GB/s':>8} {'SOL%':>6}")
    tot=0
    for (b,s) in SHAPES:
        f = torch.randn(b,s,64, device="cuda")
        o = run(f, 1.0)
        r = ref(f, 1.0)
        ok = torch.equal(o[0], r[0]) and torch.equal(o[1], r[1])
        w = bench(run, f, 1.0)
        g = bench_gpu(run, f, 1.0)
        rf = bench(ref, f, 1.0)
        byts = b*s*64*4 + 2*b*s*128*2
        gbs = byts/(g*1e-3)/1e9
        tot += w
        print(f"{str((b,s)):>14} {b*s:>7} {byts/1e6:>7.2f} {w:>9.4f} {g:>9.4f} {rf:>9.4f} {gbs:>8.1f} {gbs/8000*100:>5.1f}% {'OK' if ok else 'MISMATCH'}")
    print(f"total wall: {tot:.4f} ms")
