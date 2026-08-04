from lt import *
import torch, triton, triton.language as tl
@triton.jit
def _rowsum(X, O, N, H, BM: tl.constexpr, NC: tl.constexpr):
    pid=tl.program_id(0)
    rows=pid*BM+tl.arange(0,BM); rm=rows<N
    base=rows[:,None,None]*H
    t=tl.arange(0,64)[None,:,None]
    c=tl.arange(0,NC)[None,None,:]
    acc0=tl.zeros([BM,64],dtype=tl.float32); acc1=tl.zeros([BM,64],dtype=tl.float32)
    acc2=tl.zeros([BM,64],dtype=tl.float32); acc3=tl.zeros([BM,64],dtype=tl.float32)
    ch=(c*64+t)*4
    m=rm[:,None,None]
    acc0=tl.sum(tl.load(X+base+ch+0,mask=m,other=0.),axis=2)
    acc1=tl.sum(tl.load(X+base+ch+1,mask=m,other=0.),axis=2)
    acc2=tl.sum(tl.load(X+base+ch+2,mask=m,other=0.),axis=2)
    acc3=tl.sum(tl.load(X+base+ch+3,mask=m,other=0.),axis=2)
    s=acc0+acc1
    s=s+acc2
    s=s+acc3
    tl.store(O+rows, tl.sum(s,axis=1), mask=rm)

B,S=8,853; H=512; N=B*S
inp=gen(B,S,seed=8853)
gn=(inp['grad_output']*inp['ln_weight']).view(N,H).contiguous()
ref=gn.sum(-1)
for BM in [1,2,4,8]:
  for nw in [1,4]:
    o=torch.empty(N,device=dev)
    _rowsum[(triton.cdiv(N,BM),)](gn,o,N,H,BM=BM,NC=2,num_warps=nw,enable_fp_fusion=False)
    print(f"BM={BM} nw={nw} bitexact={torch.equal(o,ref)} frac={(o==ref).float().mean().item():.4f}")
