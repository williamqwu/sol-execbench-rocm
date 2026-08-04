import torch, torch.nn.functional as F
DEV="cuda:0"; H=2048
torch.manual_seed(0)
B,S=1,32
Bx=torch.randn(B,H,S,dtype=torch.bfloat16,device=DEV)
cw=torch.randn(H,1,4,dtype=torch.bfloat16,device=DEV)
cb=torch.randn(H,dtype=torch.bfloat16,device=DEV)
Bxp=F.pad(Bx,(3,0))
conv=F.conv1d(Bxp,cw,cb,groups=H)
print("conv dtype", conv.dtype)

exact=torch.zeros(B,H,S,device=DEV,dtype=torch.float64)
for k in range(4): exact+=Bxp[:,:,k:k+S].double()*cw[:,0,k].double()[None,:,None]
exact+=cb.double()[None,:,None]

rne = exact.to(torch.bfloat16)
# truncation to bf16 from fp32
e32 = exact.float()
bits = e32.view(torch.int32)
trunc = ((bits >> 16) << 16).view(torch.float32).to(torch.bfloat16)
print("conv==RNE(exact):", torch.equal(conv, rne), (conv!=rne).float().mean().item())
print("conv==TRUNC(exact):", torch.equal(conv, trunc), (conv!=trunc).float().mean().item())

# Is conv maybe fp32 output of exact stored... no it's bf16.
d = (conv.float()-rne.float())
print("signed diff unique-ish:", d.flatten()[:20].tolist())
print("exact:", exact.flatten()[:6].tolist())
print("conv :", conv.float().flatten()[:6].tolist())
print("rne  :", rne.float().flatten()[:6].tolist())

# hypothesis: conv computed with bf16 rounding of partial sums (accum in bf16)
a=torch.zeros(B,H,S,device=DEV,dtype=torch.float32)
for k in range(4):
    a = (a + Bxp[:,:,k:k+S].float()*cw[:,0,k].float()[None,:,None])
    a = a.to(torch.bfloat16).float()
a=(a+cb.float()[None,:,None]).to(torch.bfloat16)
print("conv==bf16-rounded-partials:", torch.equal(conv,a), (conv!=a).float().mean().item())
