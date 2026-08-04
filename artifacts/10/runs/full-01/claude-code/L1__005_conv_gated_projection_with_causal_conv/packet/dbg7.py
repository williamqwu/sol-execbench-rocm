import torch, torch.nn.functional as F, itertools
DEV="cuda:0"; H=2048
torch.manual_seed(0)
B,S=1,256
Bx=torch.randn(B,H,S,dtype=torch.bfloat16,device=DEV)
cw=torch.randn(H,1,4,dtype=torch.bfloat16,device=DEV)
cb=torch.randn(H,dtype=torch.bfloat16,device=DEV)
Bxp=F.pad(Bx,(3,0))
conv=F.conv1d(Bxp,cw,cb,groups=H)

def bf(t): return t.to(torch.bfloat16).float()

Xf=[Bxp[:,:,k:k+S].float() for k in range(4)]
Wf=[cw[:,0,k].float()[None,:,None] for k in range(4)]
Cf=cb.float()[None,:,None]

def score(r):
    return (conv!=r.to(torch.bfloat16)).float().mean().item()

best=[]
for prod_bf in [0,1]:
    for acc_bf in [0,1]:
        for bias_pos in ["first","last"]:
            for order in itertools.permutations(range(4)):
                a = Cf.expand(B,H,S).clone() if bias_pos=="first" else torch.zeros(B,H,S,device=DEV)
                for k in order:
                    p = Xf[k]*Wf[k]
                    if prod_bf: p = bf(p)
                    a = a + p
                    if acc_bf: a = bf(a)
                if bias_pos=="last":
                    a = a + Cf
                    if acc_bf: a = bf(a)
                best.append((score(a), prod_bf, acc_bf, bias_pos, order))
best.sort()
for s,pb,ab,bp,o in best[:8]:
    print(f"frac_diff={s:.6f}  prod_bf16={pb} acc_bf16={ab} bias={bp} order={o}")
