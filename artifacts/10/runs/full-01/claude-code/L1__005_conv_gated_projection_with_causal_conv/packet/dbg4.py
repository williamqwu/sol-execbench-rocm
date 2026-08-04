import torch, torch.nn.functional as F, itertools
DEV="cuda:0"; H=2048
torch.manual_seed(0)
B,S=1,64
Bx=torch.randn(B,H,S,dtype=torch.bfloat16,device=DEV)
cw=torch.randn(H,1,4,dtype=torch.bfloat16,device=DEV)
cb=torch.randn(H,dtype=torch.bfloat16,device=DEV)
Bxp=F.pad(Bx,(3,0))
conv=F.conv1d(Bxp,cw,cb,groups=H)
conv2=F.conv1d(Bxp,cw,cb,groups=H)
print("deterministic:", torch.equal(conv,conv2))

def rep(name, r):
    r=r.to(torch.bfloat16)
    print(f"{name:38s} frac_diff={(conv!=r).float().mean().item():.5f}")

def build(acc_dtype, prod_bf16, bias_first, order):
    a = torch.zeros(B,H,S,device=DEV,dtype=acc_dtype)
    if bias_first: a = a + cb.to(acc_dtype)[None,:,None]
    for k in order:
        p = Bxp[:,:,k:k+S].float()*cw[:,0,k].float()[None,:,None]
        if prod_bf16: p = p.to(torch.bfloat16)
        a = (a + p.to(acc_dtype)).to(acc_dtype)
    if not bias_first: a = a + cb.to(acc_dtype)[None,:,None]
    return a

for acc in [torch.float32, torch.bfloat16, torch.float16]:
    for pb in [False, True]:
        for bf in [False, True]:
            for order in [(0,1,2,3),(3,2,1,0)]:
                r=build(acc,pb,bf,order)
                d=(conv!=r.to(torch.bfloat16)).float().mean().item()
                if d < 0.05:
                    print(f"CLOSE acc={acc} prodbf16={pb} biasfirst={bf} order={order} frac={d:.5f}")
print("---- fp16 accum full ----")
rep("fp16 acc fwd bias last", build(torch.float16,False,False,(0,1,2,3)))
rep("fp16 acc fwd bias first", build(torch.float16,False,True,(0,1,2,3)))
rep("bf16 acc fwd bias last", build(torch.bfloat16,False,False,(0,1,2,3)))
rep("bf16 acc fwd bias first", build(torch.bfloat16,False,True,(0,1,2,3)))
rep("fp32 exact", build(torch.float32,False,False,(0,1,2,3)))

# unfold + matmul path
u = Bxp.unfold(2,4,1)  # B,H,S,4
r = (u.float()*cw[:,0,:].float()[None,:,None,:]).sum(-1)+cb.float()[None,:,None]
rep("unfold+sum fp32", r)
# matmul per channel via einsum bf16
r2 = torch.einsum('bhsk,hk->bhs', u.float(), cw[:,0,:].float())+cb.float()[None,:,None]
rep("einsum fp32", r2)
