import torch, triton, triton.language as tl, importlib, sys
DEV="cuda:0"; H=3072; TOPK=8
MS=[131,256,512,1024,2048,2164,3758,4096,8192]

def make(M, seed=0):
    g=torch.Generator(device=DEV); g.manual_seed(seed)
    return (torch.randn(M,H,dtype=torch.bfloat16,device=DEV,generator=g),
            torch.randn(M*TOPK,H,dtype=torch.bfloat16,device=DEV,generator=g),
            torch.randint(0,M,(M*TOPK,),dtype=torch.long,device=DEV,generator=g))

def ref(f,s,i):
    o=f.clone(); o.index_add_(0,i,s); return o

def bench_steady(fn,args,K=20,reps=8):
    for _ in range(5): fn(*args)
    torch.cuda.synchronize()
    best=1e9
    for _ in range(reps):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record()
        for _ in range(K): fn(*args)
        e.record(); torch.cuda.synchronize()
        best=min(best, s.elapsed_time(e)*1000.0/K)
    return best

def sol_us(M): return (10*M*H*2 + TOPK*M*8)/5.6e12*1e6

def check(o,r,atol=0.234375,rtol=0.375):
    o=o.float(); r=r.float(); d=(o-r).abs()
    return (((d<=atol)|(d<=rtol*r.abs())).float().mean().item())

if __name__=="__main__":
    cands={"ref":ref}
    for m in sys.argv[1:]:
        cands[m]=importlib.import_module(m).run
    print(f"{'M':>6} {'SOL':>7} "+" ".join(f"{k:>11}" for k in cands))
    tot={k:0.0 for k in cands}
    for M in MS:
        f,s,i=make(M); r=ref(f,s,i); row=[]
        for k,fn in cands.items():
            o=fn(f,s,i); rr=check(o,r)
            t=bench_steady(fn,(f,s,i)); tot[k]+=t
            row.append(f"{t:11.1f}" if rr>=0.99 else f"  BAD{rr:.3f}")
        print(f"{M:6d} {sol_us(M):7.1f} "+" ".join(row))
    print("TOTAL  "+"        "+" ".join(f"{tot[k]:11.1f}" for k in cands))
