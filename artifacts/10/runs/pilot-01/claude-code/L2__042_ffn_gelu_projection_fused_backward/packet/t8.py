from lt import *
import torch, kernel, triton, reference
B,S=8,853; H=512; I=2048; N=B*S
inp=gen(B,S,seed=B*1000+S)
atol=2.709629685733914e-06; rtol=1.1920928955078125e-07
go=inp['grad_output']; norm=inp['normalized']; var=inp['var']; lnw=inp['ln_weight']; eps=inp['eps']
gn=go*lnw; std=torch.sqrt(var+eps)
m1=gn.mean(-1,keepdim=True); m2=(gn*norm).mean(-1,keepdim=True)
gro_ref=((1.0/std)*(gn-m1-norm*m2))
f2b_ref=gro_ref.sum(dim=(0,1))

go2=go.reshape(N,H).contiguous(); nm2=norm.reshape(N,H).contiguous(); var1=var.reshape(N)
def runln(BM,nw):
    nb=triton.cdiv(N,BM); npg=min(nb,2048)
    gro=torch.empty((N,H),device=dev)
    pl=torch.empty((npg,H),device=dev); pb=torch.empty((npg,H),device=dev); pf=torch.empty((npg,H),device=dev)
    kernel._ln_bwd[(npg,)](go2,nm2,var1,lnw,gro,pl,pb,pf,N,H,eps,BLOCK_M=BM,BLOCK_H=512,num_warps=nw,num_stages=2)
    return gro
for BM,nw in [(8,4),(4,4),(1,4),(16,8)]:
    gro=runln(BM,nw)
    e=(gro-gro_ref.view(N,H)).abs(); t=atol+rtol*gro_ref.abs().view(N,H)
    f2b=gro.sum(0); e2=(f2b-f2b_ref).abs(); t2=atol+rtol*f2b_ref.abs()
    print(f"BM={BM} nw={nw} gro_bitexact={torch.equal(gro,gro_ref.view(N,H))} gro_m={(e<=t).float().mean().item():.6f} f2b_m={(e2<=t2).float().mean().item():.4f} f2b_maxerr={e2.max().item():.3e}")
# torch-exact version reduction check
print("torch f2b view sum0 bitexact:", torch.equal(gro_ref.view(N,H).sum(0), f2b_ref))
