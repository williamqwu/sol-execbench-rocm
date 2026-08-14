import torch, torch.nn.functional as F
DEV="cuda:0"; H=2048
torch.manual_seed(0)
B,S=2,2048; M=B*S
x=torch.randn(B,S,H,dtype=torch.bfloat16,device=DEV)
w1=torch.randn(3*H,H,dtype=torch.bfloat16,device=DEV)
b1=torch.randn(3*H,dtype=torch.bfloat16,device=DEV)
cw=torch.randn(H,1,4,dtype=torch.bfloat16,device=DEV)
cb=torch.randn(H,dtype=torch.bfloat16,device=DEV)

bcx=F.linear(x.reshape(M,H),w1,b1)
BCx=bcx.view(B,S,3*H).transpose(-1,-2)
Bc,Cc,xp=BCx.chunk(3,dim=1)
Bx=Bc*xp
print("Bx strides",Bx.stride(),"contig",Bx.is_contiguous(), "shape",Bx.shape)
Bxp=F.pad(Bx,(3,0))
print("Bxp strides",Bxp.stride(),"contig",Bxp.is_contiguous())
conv=F.conv1d(Bxp,cw,cb,groups=H)
print("conv strides",conv.stride(),"contig",conv.is_contiguous())

# Does a freshly-built contiguous padded tensor give the same conv result?
alt=torch.zeros(B,H,S+3,dtype=torch.bfloat16,device=DEV)
alt[:,:,3:]=Bx
print("alt equals Bxp:", torch.equal(alt,Bxp))
conv_alt=F.conv1d(alt,cw,cb,groups=H)
print("conv(alt)==conv(Bxp):", torch.equal(conv_alt,conv))

# is conv result layout-dependent on input being a pad output vs plain?
Bx_c = Bx.contiguous()
print("Bx.contiguous() equals Bx bitwise:", torch.equal(Bx_c,Bx))
conv_c=F.conv1d(F.pad(Bx_c,(3,0)),cw,cb,groups=H)
print("conv via contiguous Bx:", torch.equal(conv_c,conv))

# timing
def bench(fn,n=50):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    st=torch.cuda.Event(True);en=torch.cuda.Event(True);st.record()
    for _ in range(n): fn()
    en.record();torch.cuda.synchronize();return st.elapsed_time(en)/n*1000
print()
print("conv1d alone      : %.1f us"%bench(lambda: F.conv1d(Bxp,cw,cb,groups=H)))
print("pad alone         : %.1f us"%bench(lambda: F.pad(Bx,(3,0))))
print("Bc*xp             : %.1f us"%bench(lambda: Bc*xp))
print("Cc*conv           : %.1f us"%bench(lambda: Cc*conv))
print("transpose+contig  : %.1f us"%bench(lambda: (Cc*conv).transpose(-1,-2).contiguous()))
print("gemm1             : %.1f us"%bench(lambda: F.linear(x.reshape(M,H),w1,b1)))
