from lt import *
import torch, triton, triton.language as tl
@triton.jit
def kmean(GN,M1,N,H,BM: tl.constexpr,BH: tl.constexpr):
    pid=tl.program_id(0); cols=tl.arange(0,BH); cm=cols<H
    rows=pid*BM+tl.arange(0,BM); rm=rows<N
    gn=tl.load(GN+rows[:,None]*H+cols[None,:],mask=rm[:,None]&cm[None,:],other=0.)
    tl.store(M1+rows, tl.sum(gn,axis=1)/H, mask=rm)
B,S=8,853; H=512; N=B*S
inp=gen(B,S,seed=B*1000+S)
go=inp['grad_output']; lnw=inp['ln_weight']
gn=(go*lnw).view(N,H).contiguous()
m1_ref=gn.mean(-1)
m1s_ref=gn.sum(-1)/H
print("torch mean == sum/H:", torch.equal(m1_ref, m1s_ref))
for BM in [1,4,8]:
  for nw in [1,2,4,8]:
    o=torch.empty(N,device=dev)
    kmean[(triton.cdiv(N,BM),)](gn,o,N,H,BM=BM,BH=512,num_warps=nw,enable_fp_fusion=False)
    print(f" BM={BM} nw={nw} mean_bitexact={torch.equal(o,m1_ref)} fd={(o!=m1_ref).float().mean().item():.4f}")
