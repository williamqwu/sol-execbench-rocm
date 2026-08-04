from lt import *
import reference, torch
B,S=32,4096; H=512; I=2048; BS=B*S
inp=gen(B,S)
go=inp['grad_output'].view(BS,H); hs=inp['hidden_states'].view(BS,H); f1w=inp['fc1_weight']
f1o=inp['fc1_output'].view(BS,I); gout=inp['gelu_output'].view(BS,I); f2w=inp['fc2_weight']
norm=inp['normalized'].view(BS,H); var=inp['var'].view(BS,1); lnw=inp['ln_weight']; eps=inp['eps']
gro = torch.empty(BS,H,device=dev); gg=torch.empty(BS,I,device=dev)
gfcw2=torch.empty(H,I,device=dev); gfcw1=torch.empty(I,H,device=dev); ghs=torch.empty(BS,H,device=dev)

def part_ln():
    glw=(inp['grad_output']*inp['normalized']).sum(dim=(0,1))
    glb=inp['grad_output'].sum(dim=(0,1))
    gn=go*lnw
    std=torch.sqrt(var+eps)
    m1=gn.mean(-1,keepdim=True); m2=(gn*norm).mean(-1,keepdim=True)
    return (1.0/std)*(gn-m1-norm*m2)
print("ln section   ", bench(part_ln))
print("  gfc2bias   ", bench(lambda: gro.sum(0)))
print("gemm gfc2w   ", bench(lambda: torch.mm(gro.t(),gout,out=gfcw2)))
print("gemm ggelu   ", bench(lambda: torch.mm(gro,f2w,out=gg)))
def gelubwd():
    x=f1o; sp=0.7978845608028654; c=0.044715
    t=torch.tanh(sp*(x+c*x*x*x))
    return gg*(0.5*(1.0+t)+0.5*x*(1.0-t*t)*(sp*(1.0+3.0*c*x*x)))
print("gelu bwd     ", bench(gelubwd))
print("gfc1bias     ", bench(lambda: gg.sum(0)))
print("gemm gfc1w   ", bench(lambda: torch.mm(gg.t(),hs,out=gfcw1)))
print("gemm ghs     ", bench(lambda: torch.mm(gg,f1w,out=ghs)))
print("add          ", bench(lambda: ghs+gro))
