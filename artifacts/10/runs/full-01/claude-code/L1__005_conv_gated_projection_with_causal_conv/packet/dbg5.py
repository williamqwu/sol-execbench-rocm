import torch, torch.nn.functional as F
DEV="cuda:0"; H=2048
torch.manual_seed(0)
B,S=1,16
Bx=torch.randn(B,H,S,dtype=torch.bfloat16,device=DEV)
cw=torch.randn(H,1,4,dtype=torch.bfloat16,device=DEV)
cb=torch.randn(H,dtype=torch.bfloat16,device=DEV)
Bxp=F.pad(Bx,(3,0))
conv_gpu=F.conv1d(Bxp,cw,cb,groups=H).cpu()

# CPU reference in float64 from CPU copies
Bxp_c=Bxp.cpu().double(); cw_c=cw.cpu().double(); cb_c=cb.cpu().double()
ex=torch.zeros(B,H,S,dtype=torch.float64)
for k in range(4): ex+=Bxp_c[:,:,k:k+S]*cw_c[:,0,k][None,:,None]
ex+=cb_c[None,:,None]
rne=ex.to(torch.bfloat16)
print("gpu conv == RNE(exact):", torch.equal(conv_gpu,rne), (conv_gpu!=rne).float().mean().item())

# CPU conv1d itself
conv_cpu=F.conv1d(Bxp.cpu(),cw.cpu(),cb.cpu(),groups=H)
print("cpu conv == RNE(exact):", torch.equal(conv_cpu,rne), (conv_cpu!=rne).float().mean().item())
print("cpu conv == gpu conv  :", torch.equal(conv_cpu,conv_gpu), (conv_cpu!=conv_gpu).float().mean().item())

# Look at a differing element in detail
diff=(conv_gpu!=rne).nonzero()
print("n diff", diff.shape)
for i in range(3):
    b,h,s = diff[i].tolist()
    ins = Bxp_c[b,h,s:s+4].tolist()
    ws  = cw_c[h,0,:].tolist()
    print(f"--- b={b} h={h} s={s}")
    print("  in ", ins)
    print("  w  ", ws)
    print("  bias", cb_c[h].item())
    print("  exact", ex[b,h,s].item())
    print("  gpu  ", conv_gpu[b,h,s].float().item())
    print("  rne  ", rne[b,h,s].float().item())
    print("  cpu  ", conv_cpu[b,h,s].float().item())
