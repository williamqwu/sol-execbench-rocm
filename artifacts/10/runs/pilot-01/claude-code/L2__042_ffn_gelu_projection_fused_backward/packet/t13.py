from lt import *
import torch, triton, triton.language as tl
# match torch order: gn=go*lnw; m1=mean(gn); m2=mean(gn*norm); (1/std)*(gn-m1-norm*m2)
@triton.jit
def kln(GO,NM,VAR,LNW,GRO,N,H,eps,BM: tl.constexpr,BH: tl.constexpr):
    pid=tl.program_id(0)
    cols=tl.arange(0,BH); cm=cols<H
    lnw=tl.load(LNW+cols,mask=cm,other=0.)
    rows=pid*BM+tl.arange(0,BM); rm=rows<N
    off=rows[:,None]*H+cols[None,:]; m2d=rm[:,None]&cm[None,:]
    go=tl.load(GO+off,mask=m2d,other=0.); nm=tl.load(NM+off,mask=m2d,other=0.)
    var=tl.load(VAR+rows,mask=rm,other=0.)
    gn=go*lnw
    m1=tl.sum(gn,axis=1)/H
    mm=tl.sum(gn*nm,axis=1)/H
    std=tl.sqrt(var+eps)
    gro=(1.0/std)[:,None]*(gn-m1[:,None]-nm*mm[:,None])
    tl.store(GRO+off,gro,mask=m2d)

B,S=8,853; H=512; N=B*S
inp=gen(B,S,seed=B*1000+S)
go=inp['grad_output']; norm=inp['normalized']; var=inp['var']; lnw=inp['ln_weight']; eps=inp['eps']
gn=go*lnw; std=torch.sqrt(var+eps)
m1=gn.mean(-1,keepdim=True); m2=(gn*norm).mean(-1,keepdim=True)
gro_ref=((1.0/std)*(gn-m1-norm*m2)).view(N,H)
go2=go.view(N,H).contiguous(); nm2=norm.view(N,H).contiguous(); var1=var.view(N).contiguous()
for BM in [1,2,4,8]:
  for nw in [4,8]:
    gro=torch.empty((N,H),device=dev)
    kln[(triton.cdiv(N,BM),)](go2,nm2,var1,lnw,gro,N,H,eps,BM=BM,BH=512,num_warps=nw,enable_fp_fusion=False)
    print(f"BM={BM} nw={nw} bitexact={torch.equal(gro,gro_ref)} fracdiff={(gro!=gro_ref).float().mean().item():.6f} maxerr={(gro-gro_ref).abs().max().item():.3e}")
