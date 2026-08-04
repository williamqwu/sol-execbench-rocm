import torch, sys, enum
if not hasattr(enum,'StrEnum'):
    class StrEnum(str, enum.Enum):
        def __str__(self): return self.value
    enum.StrEnum = StrEnum
sys.path.insert(0,'.')
import reference as R, kernel
dev='cuda:0'
print("allow_tf32", torch.backends.cuda.matmul.allow_tf32, "fp32_prec", torch.backends.cuda.matmul.fp32_precision if hasattr(torch.backends.cuda.matmul,'fp32_precision') else '?')
torch.manual_seed(0)
M,K,N=256,7680,7680
a=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)
b=torch.randn(N,device=dev,dtype=torch.bfloat16)

# replicate reference quantization exactly
E=448.0
af=a.float()
sx=(af.reshape(M,1,K//128,128).abs().amax(3).amax(1)/E).clamp(min=1e-12)  # (M,K/128)
qx=torch.clamp(af.reshape(M,1,K//128,128)/sx[:,None,:,None],-E,E).reshape(M,K).to(torch.float8_e4m3fn)
wt=w.float().T  # (K,N)
sw=(wt.reshape(K//128,128,N//128,128).abs().amax(3).amax(1)/E).clamp(min=1e-12) # (K/128,N/128)
qw=torch.clamp(wt.reshape(K//128,128,N//128,128)/sw[:,None,:,None],-E,E).reshape(K,N).T.to(torch.float8_e4m3fn)

ref=R.run(a,w,b)
# gold in float64
adq=(qx.float().reshape(M,1,K//128,128)*sx[:,None,:,None]).reshape(M,K).double()
swc=sw.T.contiguous() # (N/128,K/128)
bdq=(qw.float().reshape(N//128,128,K//128,128)*swc[:,None,:,None]).reshape(N,K).double()
gold=(adq@bdq.T + b.double())
print("gold vs ref(f32 path) maxerr:", (ref.float().double()-gold).abs().max().item())
print("gold->bf16 vs ref bit-exact frac:", (gold.to(torch.bfloat16)==ref).float().mean().item())
out=kernel.run(a,w,b)
print("gold vs kernel maxerr:", (out.float().double()-gold).abs().max().item())
print("gold->bf16 vs kernel bit-exact frac:", (gold.to(torch.bfloat16)==out).float().mean().item())
