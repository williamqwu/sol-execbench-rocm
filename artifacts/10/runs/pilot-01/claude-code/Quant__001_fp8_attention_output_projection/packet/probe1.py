import torch, triton, triton.language as tl
print("torch", torch.__version__, "triton", triton.__version__)
dev = "cuda:0"
props = torch.cuda.get_device_properties(0)
print(props)

# 1) fp8 dot in triton?
@triton.jit
def k_dot(a_ptr, b_ptr, c_ptr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a = tl.load(a_ptr + offs_m[:, None]*K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] + offs_n[None, :]*K)
    acc = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(c_ptr + offs_m[:, None]*BN + offs_n[None, :], acc)

M=N=K=128
a = (torch.randn(M,K,device=dev)*3).to(torch.float8_e4m3fn)
b = (torch.randn(N,K,device=dev)*3).to(torch.float8_e4m3fn)
c = torch.zeros(M,N,device=dev,dtype=torch.float32)
try:
    k_dot[(1,)](a,b,c,K,M,N,K)
    ref = a.to(torch.float32) @ b.to(torch.float32).T
    print("fp8 dot OK, err", (c-ref).abs().max().item())
except Exception as e:
    print("fp8 dot FAIL", type(e).__name__, str(e)[:2000])
