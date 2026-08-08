import sys, torch, triton, triton.language as tl
sys.path.insert(0,"/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear")
import reference as R, tk
dev='cuda'; torch.manual_seed(0)
nt=512
inp=R.get_inputs({"num_tokens":nt},dev)
hs,guw=inp["hidden_states"],inp["gate_up_weight"]
asc_ref=R.BlockwiseScaler(R.ScalingType.BlockWise1x128); wsc_ref=R.BlockwiseScaler(R.ScalingType.BlockWise128x128)
h32=hs.float(); s_h=asc_ref.compute_scales(h32); h_fp8=asc_ref.apply_scaling(h32,s_h,False,True).to(torch.float8_e4m3fn)
gw_t=guw.float().T; s_gu=wsc_ref.compute_scales(gw_t); gu_fp8=wsc_ref.apply_scaling(gw_t,s_gu,False,True).T.to(torch.float8_e4m3fn)
gemm=R.CuBLASRefBlockwiseGemm()
ref=gemm.scaled_mm(h_fp8,gu_fp8,s_h,R.ScalingType.BlockWise1x128,s_gu.T.contiguous(),R.ScalingType.BlockWise128x128,None,torch.bfloat16,True)

E=tl.constexpr(448.0)
@triton.jit
def g1(A,SA,W,SW,C,M,K,N,sam,ssam,swn,sswn,scm,BLOCK_M:tl.constexpr,NUM_K:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1)
    rm=pm*BLOCK_M+tl.arange(0,BLOCK_M); rn=pn*128+tl.arange(0,128); rk=tl.arange(0,128)
    ap=A+rm[:,None]*sam+rk[None,:]; bp=W+rn[:,None]*swn+rk[None,:]
    acc=tl.zeros((BLOCK_M,128),dtype=tl.float32)
    for kb in tl.range(0,NUM_K):
        a=tl.load(ap); b=tl.load(bp)
        sa=tl.load(SA+rm*ssam+kb); sb=tl.load(SW+pn*sswn+kb)
        acc+=tl.dot(a,tl.trans(b))*(sa[:,None]*sb)
        ap+=128; bp+=128
    tl.store(C+rm[:,None]*scm+rn[None,:],acc.to(tl.bfloat16))

aq,asc=tk.quant_act(hs); wq,wsc=tk.quant_weight(guw)
print("aq==h_fp8", (aq.view(torch.uint8)==h_fp8.view(torch.uint8)).float().mean().item())
M=nt;K=3584;N=4096
C=torch.empty((M,N),dtype=torch.bfloat16,device=dev)
g1[(M//128,N//128)](aq,asc,wq,wsc,C,M,K,N,aq.stride(0),asc.stride(0),wq.stride(0),wsc.stride(0),C.stride(0),BLOCK_M=128,NUM_K=K//128,num_warps=8)
d=(C.float()-ref.float()).abs()
print("gemm1 maxabs",d.max().item(),"mean",d.mean().item(),"refmax",ref.float().abs().max().item())
print("exact frac", (C.view(torch.int16)==ref.view(torch.int16)).float().mean().item())
