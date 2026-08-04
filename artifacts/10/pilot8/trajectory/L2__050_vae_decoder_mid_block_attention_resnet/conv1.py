import torch, triton, triton.language as tl, time, torch.nn.functional as F
dev='cuda'

@triton.jit
def conv3x3(Xp, W2h, W2m, Bias, Out,
            C:tl.constexpr, Hp, Wp, H, W, HW, N,
            BM:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr, GM:tl.constexpr):
    pid=tl.program_id(0)
    nm=C//BM; nn=tl.cdiv(N,BN)
    ng=GM*nn; gid=pid//ng; fm=gid*GM; gs=min(nm-fm,GM)
    pid_m=fm+((pid%ng)%gs); pid_n=(pid%ng)//gs
    rm=pid_m*BM+tl.arange(0,BM)
    rn=pid_n*BN+tl.arange(0,BN)
    mn=rn<N
    rnc=tl.where(mn,rn,0)
    bb=rnc//HW; hw=rnc%HW; hh=hw//W; ww=hw%W
    HpWp=Hp*Wp
    base=bb*(C*HpWp)+hh*Wp+ww
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    acl=tl.zeros((BM,BN),dtype=tl.float32)
    rk=tl.arange(0,BK)
    for khw in range(9):
        koff=(khw//3)*Wp+(khw%3)
        wbase=W2h+khw*(C*C)+rm[:,None]*C
        wbasem=W2m+khw*(C*C)+rm[:,None]*C
        xb=Xp+base[None,:]+koff
        for c0 in range(0,C,BK):
            ci=c0+rk
            x=tl.load(xb+ci[:,None]*HpWp, mask=mn[None,:], other=0.)
            ah=tl.load(wbase+ci[None,:]); am=tl.load(wbasem+ci[None,:])
            xh=x.to(tl.float16)
            xm=((x-xh.to(tl.float32))*4096.0).to(tl.float16)
            acc=tl.dot(ah,xh,acc)
            acl=tl.dot(ah,xm,acl)
            acl=tl.dot(am,xh,acl)
    o=acc+acl*(1.0/4096.0)+tl.load(Bias+rm)[:,None]
    ob=bb*(C*HW)+rm[:,None]*HW+hh*W+ww
    tl.store(Out+ob, o, mask=mn[None,:])

def prep_w(w):
    C=w.shape[0]
    w2=w.permute(2,3,0,1).contiguous()  # [3,3,Co,Ci] -> khw
    w2=w2.view(9,C,C)
    wh=w2.to(torch.float16)
    wm=((w2-wh.float())*4096.0).to(torch.float16)
    return wh.contiguous(), wm.contiguous()

def conv(x, wh, wm, bias, BM=128,BN=128,BK=32,GM=8,nw=4,ns=2):
    B,C,H,W=x.shape
    xp=F.pad(x,(1,1,1,1))
    Hp,Wp=H+2,W+2
    N=B*H*W
    out=torch.empty(B,C,H,W,device=x.device,dtype=torch.float32)
    grid=(C//BM*triton.cdiv(N,BN),)
    conv3x3[grid](xp,wh,wm,bias,out,C,Hp,Wp,H,W,H*W,N,BM,BN,BK,GM,num_warps=nw,num_stages=ns)
    return out

def bench(f,n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000

if __name__=="__main__":
    import itertools
    torch.manual_seed(0)
    for (B,H,W) in [(32,32,32),(2,64,64),(1,32,32)]:
        C=512
        x=torch.randn(B,C,H,W,device=dev); w=torch.randn(C,C,3,3,device=dev)/1.732; bi=torch.randn(C,device=dev)
        ref=F.conv2d(x,w,bi,padding=1)
        ref64=F.conv2d(x.double(),w.double(),bi.double(),padding=1)
        wh,wm=prep_w(w)
        tt=bench(lambda: F.conv2d(x,w,bi,padding=1))
        e32=((ref.double()-ref64).abs()/ref64.abs().clamp(min=1e-2)).median().item()
        print(f"=== B{B} {H}x{W} torch={tt:.3f} fp32err={e32:.2e}")
        best=None
        for BM,BN,BK,nw,ns,gm in itertools.product([64,128,256],[64,128,256],[32,64],[4,8],[1,2],[4,8]):
            if BM*BN>256*128 or C%BM: continue
            try:
                o=conv(x,wh,wm,bi,BM,BN,BK,gm,nw,ns)
                t=bench(lambda: conv(x,wh,wm,bi,BM,BN,BK,gm,nw,ns),10)
            except Exception as ex: continue
            e=((o.double()-ref64).abs()/ref64.abs().clamp(min=1e-2)).median().item()
            if best is None or t<best[0]: best=(t,BM,BN,BK,nw,ns,gm,e)
        print("  BEST",best)
