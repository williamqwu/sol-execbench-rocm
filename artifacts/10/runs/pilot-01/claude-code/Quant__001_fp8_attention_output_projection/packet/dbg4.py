import torch, sys, enum
if not hasattr(enum,'StrEnum'):
    class S(str,enum.Enum):
        def __str__(self): return self.value
    enum.StrEnum=S
sys.path.insert(0,'.')
import reference as R, kernel_dbg
dev='cuda:0'; E=448.0
torch.manual_seed(0)
M,K,N=256,7680,7680
a=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)
b=torch.randn(N,device=dev,dtype=torch.bfloat16)
ref=R.run(a,w,b).float()
tol=0.011008+0.0078125*ref.abs()
for mode,name in [(0,"fp8 dot"),(1,"fp32 dot"),(2,"bf16 dot")]:
    o=kernel_dbg.run(a,w,b,MODE=mode).float()
    err=(o-ref).abs()
    print(f"MODE {mode} {name:9s}: matched={(err<=tol).float().mean().item():.6f} maxabs={err.max().item():.4f}")
