import torch, torch.nn.functional as F
DEV="cuda:0"; H=2048
torch.manual_seed(0)

def test(B,S,label):
    Bx=torch.randn(B,H,S,dtype=torch.bfloat16,device=DEV)
    cw=torch.randn(H,1,4,dtype=torch.bfloat16,device=DEV)
    cb=torch.randn(H,dtype=torch.bfloat16,device=DEV)
    Bxp=F.pad(Bx,(3,0))
    conv=F.conv1d(Bxp,cw,cb,groups=H)
    ex=torch.zeros(B,H,S,dtype=torch.float64,device=DEV)
    Bd=Bxp.double(); wd=cw.double(); bd=cb.double()
    for k in range(4): ex+=Bd[:,:,k:k+S]*wd[:,0,k][None,:,None]
    ex+=bd[None,:,None]
    rne=ex.to(torch.bfloat16)
    fd=(conv!=rne).float().mean().item()
    ulp_err = ((conv.float()-rne.float()).abs()/ (rne.float().abs()+1e-30)).max().item()
    print(f"{label:22s} B={B} S={S:5d}  frac_diff_vs_exact={fd:.5f}  max_rel={ulp_err:.4f}")
    return conv, Bxp, cw, cb, rne

for (B,S) in [(1,256),(2,4096),(1,8192),(32,256),(4,541),(1,128)]:
    test(B,S,"realistic")

print()
print("=== receptive-field test (does out[s] depend on inputs outside its window?) ===")
B,S=1,256
Bx=torch.randn(B,H,S,dtype=torch.bfloat16,device=DEV)
cw=torch.randn(H,1,4,dtype=torch.bfloat16,device=DEV)
cb=torch.randn(H,dtype=torch.bfloat16,device=DEV)
Bxp=F.pad(Bx,(3,0))
c0=F.conv1d(Bxp,cw,cb,groups=H)
Bxp2=Bxp.clone()
Bxp2[0,:,100]+=1.0   # padded index 100 -> affects outputs s=97..100
c1=F.conv1d(Bxp2,cw,cb,groups=H)
ch=(c0!=c1)[0].any(0).nonzero().flatten().tolist()
print("outputs changed at s =", ch)
print("expected receptive:", [97,98,99,100])
