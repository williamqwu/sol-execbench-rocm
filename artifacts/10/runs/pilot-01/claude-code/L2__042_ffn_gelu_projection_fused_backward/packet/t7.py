from lt import *
import torch, kernel, triton, reference
B,S=8,853; H=512; I=2048; N=B*S
inp=gen(B,S,seed=B*1000+S)
atol=2.709629685733914e-06; rtol=1.1920928955078125e-07
go=inp['grad_output']; norm=inp['normalized']; var=inp['var']; lnw=inp['ln_weight']; eps=inp['eps']
# reference gro
gn=go*lnw; std=torch.sqrt(var+eps)
m1=gn.mean(-1,keepdim=True); m2=(gn*norm).mean(-1,keepdim=True)
gro_ref=((1.0/std)*(gn-m1-norm*m2)).view(N,H)

# run kernel pieces
go2=go.reshape(N,H).contiguous(); nm2=norm.reshape(N,H).contiguous(); var1=var.reshape(N)
BLOCK_H=512; BLOCK_M=8
nb=triton.cdiv(N,BLOCK_M); npg=min(nb,2048)
gro=torch.empty((N,H),device=dev); 
pl=torch.empty((npg,H),device=dev); pb=torch.empty((npg,H),device=dev); pf=torch.empty((npg,H),device=dev)
kernel._ln_bwd[(npg,)](go2,nm2,var1,lnw,gro,pl,pb,pf,N,H,eps,BLOCK_M=BLOCK_M,BLOCK_H=BLOCK_H,num_warps=4,num_stages=2)
e=(gro-gro_ref).abs(); tol=atol+rtol*gro_ref.abs()
print("gro matched",(e<=tol).float().mean().item(),"maxerr",e.max().item(),"|ref|max",gro_ref.abs().max().item())

lnw_ref=(go*norm).sum(dim=(0,1)); lnb_ref=go.sum(dim=(0,1)); f2b_ref=gro_ref.sum(0)
for nm_,p,r in [('lnw',pl,lnw_ref),('lnb',pb,lnb_ref)]:
    o=kernel._reduce(p,npg,H); e=(o-r).abs(); t=atol+rtol*r.abs()
    print(nm_,"matched",(e<=t).float().mean().item(),"maxerr",e.max().item(),"|r|max",r.abs().max().item())
