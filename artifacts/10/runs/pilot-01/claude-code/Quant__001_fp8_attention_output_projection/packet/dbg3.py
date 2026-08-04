import torch, sys, enum
if not hasattr(enum,'StrEnum'):
    class S(str,enum.Enum):
        def __str__(self): return self.value
    enum.StrEnum=S
sys.path.insert(0,'.')
import reference as R, kernel
dev='cuda:0'; E=448.0
torch.manual_seed(0)
M,K,N=256,7680,7680
a=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)
b=torch.randn(N,device=dev,dtype=torch.bfloat16)
ref=R.run(a,w,b).float()

# torch simulation of MY blockwise algorithm, fp32 accumulate per block
af=a.float(); wf=w.float()
acc=torch.zeros(M,N,device=dev,dtype=torch.float32)
for kb in range(K//128):
    ab=af[:,kb*128:(kb+1)*128]
    wb=wf[:,kb*128:(kb+1)*128]                 # (N,128)
    sa=(ab.abs().amax(1)/E).clamp(min=1e-12)   # (M,)
    qa=torch.clamp(ab/sa[:,None],-E,E).to(torch.float8_e4m3fn).float()
    swv=(wb.reshape(N//128,128,128).abs().amax(2).amax(1)/E).clamp(min=1e-12) # (N/128,)
    swb=swv.repeat_interleave(128)             # (N,)
    qw=torch.clamp(wb/swb[:,None],-E,E).to(torch.float8_e4m3fn).float()
    acc += (qa@qw.T)*(sa[:,None]*swb[None,:])
sim=(acc+b.float()).to(torch.bfloat16).float()
out=kernel.run(a,w,b).float()

def stats(name,x):
    err=(x-ref).abs(); tol=0.011008+0.0078125*ref.abs()
    print(f"{name}: matched={(err<=tol).float().mean().item():.6f} maxabs={err.max().item():.4f}")
stats("torch-sim ", sim)
stats("triton    ", out)
print("sim vs triton maxabs:", (sim-out).abs().max().item())
print("|ref| percentiles:", [round(torch.quantile(ref.abs().flatten().float(),q).item(),3) for q in (0.5,0.9,0.99,1.0)])
