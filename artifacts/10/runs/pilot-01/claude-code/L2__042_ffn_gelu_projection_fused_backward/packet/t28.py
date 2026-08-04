from lt import *
import torch, triton, triton.language as tl

@triton.jit
def _tree64(s, BM: tl.constexpr):
    # s: [BM, 64] -> [BM] via adjacent-pair balanced tree
    a = tl.sum(tl.reshape(s, (BM, 32, 2)), axis=2)
    a = tl.sum(tl.reshape(a, (BM, 16, 2)), axis=2)
    a = tl.sum(tl.reshape(a, (BM, 8, 2)), axis=2)
    a = tl.sum(tl.reshape(a, (BM, 4, 2)), axis=2)
    a = tl.sum(tl.reshape(a, (BM, 2, 2)), axis=2)
    a = tl.sum(tl.reshape(a, (BM, 1, 2)), axis=2)
    return tl.reshape(a, (BM,))

@triton.jit
def _rowsum(X, O, N, BM: tl.constexpr):
    pid=tl.program_id(0)
    rows=pid*BM+tl.arange(0,BM); rm=rows<N
    base=rows[:,None,None]*512
    t=tl.arange(0,64)[None,:,None]
    i=tl.arange(0,4)[None,None,:]
    m=rm[:,None,None]
    v0=tl.load(X+base+t*4+i, mask=m, other=0.)          # chunk c=0 -> [BM,64,4]
    v1=tl.load(X+base+256+t*4+i, mask=m, other=0.)      # chunk c=1
    acc=v0+v1                                            # [BM,64,4] acc_i per lane
    s = tl.sum(tl.reshape(acc,(BM,64,2,2)),axis=3)       # a0+a1, a2+a3 ... NO: need seq
    tl.store(O+rows, tl.zeros([BM],dtype=tl.float32), mask=rm)

# proper: sequential over 4 accumulators
@triton.jit
def _rowsum2(X, O, N, BM: tl.constexpr):
    pid=tl.program_id(0)
    rows=pid*BM+tl.arange(0,BM); rm=rows<N
    base=rows[:,None]*512
    t=tl.arange(0,64)[None,:]
    m=rm[:,None]
    a0=tl.load(X+base+t*4+0,mask=m,other=0.)+tl.load(X+base+256+t*4+0,mask=m,other=0.)
    a1=tl.load(X+base+t*4+1,mask=m,other=0.)+tl.load(X+base+256+t*4+1,mask=m,other=0.)
    a2=tl.load(X+base+t*4+2,mask=m,other=0.)+tl.load(X+base+256+t*4+2,mask=m,other=0.)
    a3=tl.load(X+base+t*4+3,mask=m,other=0.)+tl.load(X+base+256+t*4+3,mask=m,other=0.)
    s=a0+a1
    s=s+a2
    s=s+a3
    tl.store(O+rows, _tree64(s,BM), mask=rm)

B,S=8,853; H=512; N=B*S
inp=gen(B,S,seed=8853)
gn=(inp['grad_output']*inp['ln_weight']).view(N,H).contiguous()
ref=gn.sum(-1)
for BM in [1,2,4,8]:
    o=torch.empty(N,device=dev)
    _rowsum2[(triton.cdiv(N,BM),)](gn,o,N,BM=BM,num_warps=4,enable_fp_fusion=False)
    print(f"BM={BM} bitexact={torch.equal(o,ref)} frac={(o==ref).float().mean().item():.4f} maxerr={(o-ref).abs().max().item():.2e}")
