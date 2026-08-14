import torch, torch.nn.functional as F, sys
sys.path.insert(0,".")
import kernel
DEV="cuda:0"; H=2048
torch.manual_seed(0)
B,S=2,512; M=B*S
x=torch.randn(B,S,H,dtype=torch.bfloat16,device=DEV)
w1=torch.randn(3*H,H,dtype=torch.bfloat16,device=DEV)
b1=torch.randn(3*H,dtype=torch.bfloat16,device=DEV)
cw=torch.randn(H,1,4,dtype=torch.bfloat16,device=DEV)
cb=torch.randn(H,dtype=torch.bfloat16,device=DEV)

bcx = F.linear(x.reshape(M,H), w1, b1)
BCx = bcx.view(B,S,3*H).transpose(-1,-2)
Bc,Cc,xp = BCx.chunk(3,dim=1)
Bx = Bc*xp
Bxp = F.pad(Bx,(3,0))
conv = F.conv1d(Bxp, cw, cb, groups=H)
yref = (Cc*conv).transpose(-1,-2).contiguous().reshape(M,H)

ymine = kernel._mid(bcx, cw.contiguous(), cb.contiguous(), M, S, H)
d=(yref.float()-ymine.float()).abs()
print("y exact match frac:", (yref==ymine).float().mean().item())
print("y maxdiff:", d.max().item(), " ref absmax:", yref.float().abs().max().item())

# check conv stage alone in fp32 to see what conv1d does
Bx32 = (Bc.float()*xp.float())
print("Bx is bf16 rounded of fp32 product:", torch.equal(Bx.float(), Bx32.to(torch.bfloat16).float()))
# manual conv in fp32 from bf16 Bx
man = torch.zeros(B,H,S,device=DEV,dtype=torch.float32)
Bxpf = F.pad(Bx.float(),(3,0))
for k in range(4):
    man += Bxpf[:,:,k:k+S]*cw[:,0,k].float()[None,:,None]
man += cb.float()[None,:,None]
print("conv1d == manual fp32->bf16 :", torch.equal(conv.float(), man.to(torch.bfloat16).float()))
dc=(conv.float()-man.to(torch.bfloat16).float()).abs()
print("conv diff max", dc.max().item(), "frac differing", (conv!=man.to(torch.bfloat16)).float().mean().item())
