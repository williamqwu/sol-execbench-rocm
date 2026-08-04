from lt import *
import torch, triton, kernel, reference
B,S=32,4096; H=512; I=2048; N=B*S
inp=gen(B,S)
go=inp['grad_output']; norm=inp['normalized']; var=inp['var']; lnw=inp['ln_weight']; eps=inp['eps']
gelu=inp['gelu_output']; f2w=inp['fc2_weight']; f1o=inp['fc1_output']; f1w=inp['fc1_weight']; hs=inp['hidden_states']
print("TOTAL mine", bench(lambda: kernel.run(**inp)))
print("  glw+glb  ", bench(lambda: ((go*norm).sum(dim=(0,1)), go.sum(dim=(0,1)))))
def lnsec():
    gn=go*lnw; std=torch.sqrt(var+eps)
    m1=gn.mean(-1,keepdim=True); m2=(gn*norm).mean(-1,keepdim=True)
    return (1.0/std)*(gn-m1-norm*m2)
print("  lnsec    ", bench(lnsec))
gro=lnsec(); gro2=gro.view(N,H)
print("  gf2bias  ", bench(lambda: gro.sum(dim=(0,1))))
print("  GEMM f2w ", bench(lambda: gro2.t()@gelu.view(N,I)))
print("  GEMM ggel", bench(lambda: gro@f2w))
ggel=gro@f2w; gfo=torch.empty_like(ggel); n=gfo.numel()
print("  gelu tri ", bench(lambda: kernel._gelu_bwd[(triton.cdiv(n,1024),)](f1o,ggel,gfo,n,1024,enable_fp_fusion=False,num_warps=4)))
print("  gf1bias  ", bench(lambda: gfo.sum(dim=(0,1))))
print("  GEMM f1w ", bench(lambda: gfo.view(N,I).t()@hs.view(N,H)))
print("  GEMM ghs ", bench(lambda: gfo@f1w+gro))
print("  addmm ghs", bench(lambda: torch.addmm(gro2, gfo.view(N,I), f1w)))
