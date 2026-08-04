import torch, torch.nn.functional as F
DEV="cuda:0"; H=2048
torch.manual_seed(0)
B,S=1,256
Bx=torch.randn(B,H,S,dtype=torch.bfloat16,device=DEV)
cw=torch.randn(H,1,4,dtype=torch.bfloat16,device=DEV)
cb=torch.randn(H,dtype=torch.bfloat16,device=DEV)
Bxp=F.pad(Bx,(3,0))
conv=F.conv1d(Bxp,cw,cb,groups=H)

Xf=[Bxp[:,:,k:k+S].float() for k in range(4)]
Wf=[cw[:,0,k].float()[None,:,None] for k in range(4)]
Cf=cb.float()[None,:,None]
def bf(t): return t.to(torch.bfloat16).float()

# H1: products rounded to bf16, fp32 accumulate, bias last
a=torch.zeros(B,H,S,device=DEV)
for k in range(4): a=a+bf(Xf[k]*Wf[k])
h1=(a+Cf)
print("H1 prod-bf16 acc-fp32 bias-last :", (conv!=h1.to(torch.bfloat16)).float().mean().item())

# H2: same but bias first
a=Cf.expand(B,H,S).clone()
for k in range(4): a=a+bf(Xf[k]*Wf[k])
print("H2 prod-bf16 acc-fp32 bias-first:", (conv!=a.to(torch.bfloat16)).float().mean().item())

# H3: products bf16, accumulate bf16 too
a=torch.zeros(B,H,S,device=DEV)
for k in range(4): a=bf(a+bf(Xf[k]*Wf[k]))
h3=bf(a+Cf)
print("H3 prod-bf16 acc-bf16 bias-last :", (conv!=h3.to(torch.bfloat16)).float().mean().item())

# H4: bias added into accumulator as bf16 rounding each step
a=Cf.expand(B,H,S).clone()
for k in range(4): a=bf(a+bf(Xf[k]*Wf[k]))
print("H4 prod-bf16 acc-bf16 bias-first:", (conv!=a.to(torch.bfloat16)).float().mean().item())
