from lt import *
import reference, torch
B,S=32,4096; H=512; I=2048
inp=gen(B,S)
go=inp['grad_output']; norm=inp['normalized']
atol=3.3900898571863226e-06; rtol=1.1920928955078125e-07
# how sensitive is the sum reduction?
a = (go*norm).sum(dim=(0,1))
b = (go*norm).view(-1,H).sum(0)
c = (go.view(-1,H)*norm.view(-1,H)).float().sum(0)
d = torch.einsum('nh,nh->h', go.view(-1,H), norm.view(-1,H))
for nm,x in [('view sum0',b),('c',c),('einsum',d)]:
    e=(x-a).abs(); tolm=atol+rtol*a.abs()
    print(nm, "maxerr", e.max().item(), "matched", (e<=tolm).float().mean().item(), "|a|max", a.abs().max().item())
