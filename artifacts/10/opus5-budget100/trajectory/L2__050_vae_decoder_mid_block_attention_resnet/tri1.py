import torch, time
import triton, triton.language as tl

dev='cuda'
def bench(fn, iters=30, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1000

@triton.jit
def gemm3(Ah, Al, Bh, Bl, Cptr, M, N, K,
          sam, sak, sbk, sbn, scm, scn,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NSPLIT: tl.constexpr):
    pid = tl.program_id(0)
    gm = tl.cdiv(M, BM); gn = tl.cdiv(N, BN)
    GROUP = 8
    width = GROUP * gn
    gid = pid // width
    fm = gid * GROUP
    gsize = min(gm - fm, GROUP)
    pm = fm + ((pid % width) % gsize)
    pn = (pid % width) // gsize
    rm = (pm*BM + tl.arange(0,BM)) % M
    rn = (pn*BN + tl.arange(0,BN)) % N
    rk = tl.arange(0,BK)
    Ah_p = Ah + rm[:,None]*sam + rk[None,:]*sak
    Al_p = Al + rm[:,None]*sam + rk[None,:]*sak
    Bh_p = Bh + rk[:,None]*sbk + rn[None,:]*sbn
    Bl_p = Bl + rk[:,None]*sbk + rn[None,:]*sbn
    acc = tl.zeros((BM,BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K,BK)):
        m = rk[None,:] < K - k*BK
        ah = tl.load(Ah_p, mask=m, other=0.0)
        bh = tl.load(Bh_p, mask=(rk[:,None] < K-k*BK), other=0.0)
        acc += tl.dot(ah, bh)
        if NSPLIT >= 3:
            al = tl.load(Al_p, mask=m, other=0.0)
            bl = tl.load(Bl_p, mask=(rk[:,None] < K-k*BK), other=0.0)
            acc += tl.dot(ah, bl)
            acc += tl.dot(al, bh)
        Ah_p += BK*sak; Al_p += BK*sak; Bh_p += BK*sbk; Bl_p += BK*sbk
    rm2 = pm*BM + tl.arange(0,BM); rn2 = pn*BN + tl.arange(0,BN)
    tl.store(Cptr + rm2[:,None]*scm + rn2[None,:]*scn, acc,
             mask=(rm2[:,None]<M)&(rn2[None,:]<N))

def run3(a, b, nsplit=3, BM=128, BN=128, BK=64, ns=2, nw=8):
    M,K = a.shape; K2,N = b.shape
    ah = a.half(); al = (a - ah.float()).half()
    bh = b.half(); bl = (b - bh.float()).half()
    c = torch.empty((M,N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M,BM)*triton.cdiv(N,BN),)
    gemm3[grid](ah,al,bh,bl,c,M,N,K, ah.stride(0),ah.stride(1), bh.stride(0),bh.stride(1), c.stride(0),c.stride(1),
                BM=BM,BN=BN,BK=BK,NSPLIT=nsplit, num_stages=ns, num_warps=nw)
    return c, (ah,al,bh,bl)

for (M,N,K) in [(32768,512,4608),(8192,512,4608),(32768,512,512)]:
    a=torch.randn(M,K,device=dev); b=torch.randn(K,N,device=dev)
    ref = a@b
    t32 = bench(lambda: a@b)
    for nsplit in (1,3):
        c,(ah,al,bh,bl) = run3(a,b,nsplit)
        err = (c-ref).abs().max().item()/ref.abs().max().item()
        for (BM,BN,BK,ns,nw) in [(128,128,64,2,8),(256,128,64,2,8),(128,256,64,2,8),(128,128,128,2,8),(64,128,128,2,4)]:
            cc = torch.empty((M,N), device=dev, dtype=torch.float32)
            grid=(triton.cdiv(M,BM)*triton.cdiv(N,BN),)
            f = lambda: gemm3[grid](ah,al,bh,bl,cc,M,N,K, ah.stride(0),ah.stride(1),bh.stride(0),bh.stride(1),cc.stride(0),cc.stride(1),BM=BM,BN=BN,BK=BK,NSPLIT=nsplit,num_stages=ns,num_warps=nw)
            try:
                t=bench(f)
            except Exception as e:
                print("fail",e); continue
            print(f"M{M} K{K} nsplit={nsplit} {BM}x{BN}x{BK} w{nw}s{ns}: {t:.3f} ms (fp32 {t32:.3f}) relerr={err:.2e} {2*M*N*K/t*1e-9:.0f} TF/s")
