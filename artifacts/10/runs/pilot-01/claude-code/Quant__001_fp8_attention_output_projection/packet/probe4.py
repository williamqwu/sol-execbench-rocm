import torch, triton, triton.language as tl
dev='cuda:0'
@triton.jit
def k_dot(a_ptr,b_ptr,c_ptr,K: tl.constexpr,BM: tl.constexpr,BN: tl.constexpr):
    om=tl.arange(0,BM); on=tl.arange(0,BN); ok=tl.arange(0,K)
    a=tl.load(a_ptr+om[:,None]*K+ok[None,:])
    b=tl.load(b_ptr+ok[:,None]+on[None,:]*K)
    tl.store(c_ptr+om[:,None]*BN+on[None,:], tl.dot(a,b,out_dtype=tl.float32))

torch.manual_seed(0)
M=N=K=128
a=(torch.randn(M,K,device=dev)*3).to(torch.float8_e4m3fn)
b=(torch.randn(N,K,device=dev)*3).to(torch.float8_e4m3fn)
c=torch.zeros(M,N,device=dev,dtype=torch.float32)
k_dot[(1,)](a,b,c,K,M,N)
gold=(a.double()@b.double().T)
print("triton fp8 dot   vs f64:", (c.double()-gold).abs().max().item())
print("torch f32 matmul vs f64:", ((a.float()@b.float().T).double()-gold).abs().max().item())
print("gold magnitude:", gold.abs().max().item())
print("bitexact frac triton:", (c.double()==gold).float().mean().item())
# what about torch fp32 matmul precision setting
import os
print("hipblaslt allow tf32:", torch.backends.cuda.matmul.allow_tf32)
