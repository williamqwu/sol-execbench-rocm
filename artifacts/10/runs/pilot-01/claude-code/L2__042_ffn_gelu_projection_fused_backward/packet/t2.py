from lt import *
import reference, torch
B,S=32,4096; H=512; I=2048
inp=gen(B,S)
g=inp; go=g['grad_output']; hs=g['hidden_states']; f1w=g['fc1_weight']; f1o=g['fc1_output']
gout=g['gelu_output']; f2w=g['fc2_weight']; norm=g['normalized']; var=g['var']; lnw=g['ln_weight']; eps=g['eps']
BS=B*S
gro = torch.randn(BS,H,device=dev)
gg  = torch.randn(BS,I,device=dev)
print("lnw sums     ", bench(lambda: ((go*norm).sum(dim=(0,1)), go.sum(dim=(0,1)))))
print("ln bwd elemw ", bench(lambda: (go*lnw - (go*lnw).mean(-1,keepdim=True))))
print("gemm gfc2w   ", bench(lambda: gro.t()@gout.view(BS,I)))
print("gemm ggelu   ", bench(lambda: gro@f2w))
print("gelu grad ew ", bench(lambda: gg*torch.tanh(f1o)))
print("gemm gfc1w   ", bench(lambda: gg.t()@hs.view(BS,H)))
print("gemm ghs     ", bench(lambda: gg@f1w))
print("sum gg dim0  ", bench(lambda: gg.sum(0)))
print("TOTAL ref    ", bench(lambda: reference.run(**inp)))
