import torch, torch.nn.functional as F
DEV="cuda:0"; H=2048
torch.manual_seed(0)
B,S=2,2048; M=B*S
y=torch.randn(M,H,dtype=torch.bfloat16,device=DEV)*3
w2=torch.randn(H,H,dtype=torch.bfloat16,device=DEV)
b2=torch.randn(H,dtype=torch.bfloat16,device=DEV)
ref=F.linear(y,w2,b2)
atol=0.0256; rtol=0.0078125
def matched(g):
    d=(ref.float()-g.float()).abs(); thr=atol+rtol*ref.float().abs()
    return (d<=thr).float().mean().item()
print("identical:", matched(ref))
for frac in [0.01,0.03,0.1,0.3,1.0]:
    yb=y.view(torch.int16).clone()
    m=torch.rand(yb.shape,device=DEV)<frac
    # +1 ulp (magnitude direction ignoring sign subtleties)
    yb=torch.where(m, yb+1, yb)
    y2=yb.view(torch.bfloat16)
    print(f"frac={frac:5.2f} perturbed 1ulp -> matched={matched(F.linear(y2,w2,b2)):.5f}")
