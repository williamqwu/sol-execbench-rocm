import torch, time, importlib, math, sys
import reference
dev='cuda'
C=512
WL=[(1,32,32,3.8822859424124205e-06),(1,61,61,3.900360110448521e-06),(2,64,64,1.430511474609375e-04),
    (1,48,48,1.049041748046875e-04),(16,32,32,3.931760339537081e-06),(4,16,16,1.1444091796875e-04),
    (8,32,32,1.2874603271484375e-04),(32,32,32,3.964399394002429e-06),(4,48,48,1.3828277587890625e-04),
    (2,41,41,3.898631173405141e-06),(1,16,16,9.5367431640625e-05)]
RTOL=1.1920928955078125e-07
def mkargs(B,H,W,seed=0):
    torch.manual_seed(seed)
    rn=lambda *s: torch.randn(*s,device=dev)
    ones=lambda n: torch.ones(n,device=dev)
    zeros=lambda n: torch.zeros(n,device=dev)
    conv=lambda: rn(C,C,3,3)/math.sqrt(3)
    lin=lambda: rn(C,C)/math.sqrt(C)
    a=[rn(B,C,H,W), rn(B,C)]
    for p in range(2):
        a += [ones(C),zeros(C),conv(),rn(C),lin(),rn(C),ones(C),zeros(C),conv(),rn(C)]
        if p==0:
            a += [ones(C),zeros(C),lin(),rn(C),lin(),rn(C),lin(),rn(C),lin(),rn(C)]
    a.append(1e-5)
    return a
def bench(f,n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000
import kernel; importlib.reload(kernel)
only=sys.argv[1] if len(sys.argv)>1 else None
tot=1.0; n=0; nfail=0
for (B,H,W,atol) in WL:
    if only and only!=f"{B}x{H}": continue
    args=mkargs(B,H,W)
    y=reference.run(*args); x=kernel.run(*args)
    ae=(x.float()-y.float()).abs()
    bound=atol+RTOL*y.float().abs()
    ratio=1.0-(ae>bound).float().mean().item()
    tk=bench(lambda: kernel.run(*args)); tr=bench(lambda: reference.run(*args))
    sp=tr/tk; tot*=sp; n+=1
    ok = ratio>=0.99
    if not ok: nfail+=1
    print(f"B{B} {H}x{W}: match={ratio:.5f} maxae={ae.max().item():.2e} |y|max={y.abs().max().item():.1f} yours={tk:.3f} ref={tr:.3f} sp={sp:.3f} {'OK' if ok else 'FAIL'}")
print(f"geomean {tot**(1/max(n,1)):.4f}  fails={nfail}")
