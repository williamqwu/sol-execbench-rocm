from lt import *
import torch, triton, triton.language as tl, numpy as np
# Reproduce: 64 lanes, each seq-accumulates 4 accumulators over strided vec4 chunks,
# then acc0+acc1+acc2+acc3 sequentially, then butterfly tree over 64.
# For H=512, vec=4, nchunk=128, nt=64 -> each lane does 2 chunks.
# Layout per lane t: elements [t*4 .. t*4+3] and [(t+64)*4 .. +3]
# Equivalent as tensor ops: reshape row (512,) -> (2, 64, 4)  [chunk-major]
#   lane t, acc i = r[0,t,i] + r[1,t,i]
#   then v = ((a0+a1)+a2)+a3   per lane -> (64,)
#   then butterfly tree over 64 lanes
@triton.jit
def _rowsum(X, O, N, H, BM: tl.constexpr, NC: tl.constexpr):
    pid=tl.program_id(0)
    rows=pid*BM+tl.arange(0,BM); rm=rows<N
    # gather as [BM, NC, 64, 4]
    c=tl.arange(0,NC)[None,:,None,None]
    t=tl.arange(0,64)[None,None,:,None]
    i=tl.arange(0,4)[None,None,None,:]
    off=rows[:,None,None,None]*H + (c*64+t)*4 + i
    v=tl.load(X+off, mask=rm[:,None,None,None], other=0.0)
    acc=tl.sum(v,axis=1)  # [BM,64,4] -- but tl.sum order over NC
    # sequential over the 4 accumulators
    a=tl.reshape(acc,(BM,64,4))
    s=a[:,:,0]+a[:,:,1]
    s=s+a[:,:,2]
    s=s+a[:,:,3]
    # butterfly over 64
    s=tl.sum(s,axis=1)
    tl.store(O+rows, s, mask=rm)

B,S=8,853; H=512; N=B*S
inp=gen(B,S,seed=8853)
gn=(inp['grad_output']*inp['ln_weight']).view(N,H).contiguous()
ref=gn.sum(-1)
o=torch.empty(N,device=dev)
_rowsum[(triton.cdiv(N,4),)](gn,o,N,H,BM=4,NC=2,num_warps=4,enable_fp_fusion=False)
print("bitexact:",torch.equal(o,ref),"frac",(o==ref).float().mean().item(),"maxerr",(o-ref).abs().max().item())
