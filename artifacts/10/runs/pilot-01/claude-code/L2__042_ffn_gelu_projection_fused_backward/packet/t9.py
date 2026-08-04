from lt import *
import torch, kernel, triton, reference
B,S=8,853; H=512; I=2048; N=B*S
inp=gen(B,S,seed=B*1000+S)
atol=2.709629685733914e-06; rtol=1.1920928955078125e-07
def m(a,b):
    e=(a-b).abs(); t=atol+rtol*b.abs(); return (e<=t).float().mean().item(), e.max().item()
ref=reference.run(**inp)
go=inp['grad_output']; norm=inp['normalized']; var=inp['var']; lnw=inp['ln_weight']; eps=inp['eps']
gelu=inp['gelu_output']; f2w=inp['fc2_weight']; f1o=inp['fc1_output']; f1w=inp['fc1_weight']; hs=inp['hidden_states']
gn=go*lnw; std=torch.sqrt(var+eps)
m1=gn.mean(-1,keepdim=True); m2=(gn*norm).mean(-1,keepdim=True)
gro_ref=((1.0/std)*(gn-m1-norm*m2))

go2=go.reshape(N,H).contiguous(); nm2=norm.reshape(N,H).contiguous(); var1=var.reshape(N)
nb=triton.cdiv(N,8); npg=min(nb,2048)
gro=torch.empty((N,H),device=dev)
pl=torch.empty((npg,H),device=dev); pb=torch.empty((npg,H),device=dev); pf=torch.empty((npg,H),device=dev)
kernel._ln_bwd[(npg,)](go2,nm2,var1,lnw,gro,pl,pb,pf,N,H,eps,BLOCK_M=8,BLOCK_H=512,num_warps=4,num_stages=2)

print("gro           ", m(gro, gro_ref.view(N,H)))
# downstream with my gro
print("gfc2w mygro   ", m(torch.mm(gro.t(),gelu.view(N,I)), ref[3]))
print("gfc2w refgro  ", m(torch.mm(gro_ref.view(N,H).t(),gelu.view(N,I)), ref[3]))
ggel = torch.mm(gro, f2w)
ggel_r = gro_ref @ f2w
print("ggelu mygro   ", m(ggel, ggel_r.view(N,I)))
# gelu bwd triton
gfo=torch.empty((N,I),device=dev); pf1=torch.empty((triton.cdiv(N,32),I),device=dev)
kernel._gelu_bwd[(triton.cdiv(N,32),triton.cdiv(I,128))](f1o.view(N,I),ggel,gfo,pf1,N,I,BLOCK_M=32,BLOCK_I=128,num_warps=4,num_stages=2)
# ref grad_fc1_output
x=f1o; sp=0.7978845608028654; c=0.044715
t_=torch.tanh(sp*(x+c*(x*x*x)))
gg_=0.5*(1.0+t_)+0.5*x*(1.0-t_*t_)*(sp*(1.0+3.0*c*x*x))
gfo_ref=(ggel_r*gg_).view(N,I)
print("gfo           ", m(gfo, gfo_ref))
print("gfc1w         ", m(torch.mm(gfo.t(),hs.view(N,H)), ref[1]))
print("gfc1w refgfo  ", m(torch.mm(gfo_ref.t(),hs.view(N,H)), ref[1]))
print("gfc1b torchsum", m(gfo.sum(0), ref[2]))
print("gfc1b refsum  ", m(gfo_ref.sum(0), ref[2]))
print("ghs           ", m((torch.mm(gfo,f1w)+gro).view(B,S,H), ref[0]))
