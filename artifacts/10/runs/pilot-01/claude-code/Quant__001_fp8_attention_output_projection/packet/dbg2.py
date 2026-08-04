import torch, triton, triton.language as tl
dev='cuda:0'
E=448.0
FP8MAX=tl.constexpr(448.0)

@triton.jit
def dump_k(A,W,QA,SA,QW,SW,K, BM: tl.constexpr, BN: tl.constexpr):
    om=tl.arange(0,BM); on=tl.arange(0,BN); ok=tl.arange(0,128)
    a=tl.load(A+om[:,None]*K+ok[None,:]).to(tl.float32)
    w=tl.load(W+on[:,None]*K+ok[None,:]).to(tl.float32)
    sa=tl.maximum(tl.max(tl.abs(a),axis=1)/FP8MAX,1e-12)
    qa=tl.clamp(a/sa[:,None],-FP8MAX,FP8MAX).to(tl.float8e4nv)
    sw=tl.maximum(tl.max(tl.abs(w))/FP8MAX,1e-12)
    qw=tl.clamp(w/sw,-FP8MAX,FP8MAX).to(tl.float8e4nv)
    tl.store(QA+om[:,None]*128+ok[None,:], qa)
    tl.store(SA+om, sa)
    tl.store(QW+on[:,None]*128+ok[None,:], qw)
    tl.store(SW+tl.arange(0,1), sw)

torch.manual_seed(0)
M,K,N=128,7680,7680
a=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)
QA=torch.empty(M,128,device=dev,dtype=torch.float8_e4m3fn)
SA=torch.empty(M,device=dev,dtype=torch.float32)
QW=torch.empty(128,128,device=dev,dtype=torch.float8_e4m3fn)
SW=torch.empty(1,device=dev,dtype=torch.float32)
dump_k[(1,)](a,w,QA,SA,QW,SW,K,BM=M,BN=128)

# torch reference for k-block 0
af=a[:, :128].float()
sa_ref=(af.abs().amax(1)/E).clamp(min=1e-12)
qa_ref=torch.clamp(af/sa_ref[:,None],-E,E).to(torch.float8_e4m3fn)
wf=w[:128,:128].float()
sw_ref=(wf.abs().amax()/E).clamp(min=1e-12)
qw_ref=torch.clamp(wf/sw_ref,-E,E).to(torch.float8_e4m3fn)
print("sa match:", torch.equal(SA,sa_ref), (SA-sa_ref).abs().max().item())
print("qa bitexact:", (QA.view(torch.uint8)==qa_ref.view(torch.uint8)).float().mean().item())
print("sw match:", SW.item(), sw_ref.item())
print("qw bitexact:", (QW.view(torch.uint8)==qw_ref.view(torch.uint8)).float().mean().item())

# now: fp8 dot accuracy for this block
@triton.jit
def dotk(A,B,C,BM: tl.constexpr,BN: tl.constexpr):
    om=tl.arange(0,BM); on=tl.arange(0,BN); ok=tl.arange(0,128)
    x=tl.load(A+om[:,None]*128+ok[None,:])
    y=tl.load(B+on[:,None]*128+ok[None,:])
    tl.store(C+om[:,None]*BN+on[None,:], tl.dot(x,tl.trans(y),out_dtype=tl.float32))
C=torch.empty(M,128,device=dev,dtype=torch.float32)
dotk[(1,)](qa_ref,qw_ref,C,M,128)
gold=(qa_ref.double()@qw_ref.double().T)
print("fp8dot err:", (C.double()-gold).abs().max().item(), "magnitude", gold.abs().max().item())
print("fp8dot bitexact:", (C.double()==gold).float().mean().item())
tf=qa_ref.float()@qw_ref.float().T
print("torch f32 err:", (tf.double()-gold).abs().max().item())
