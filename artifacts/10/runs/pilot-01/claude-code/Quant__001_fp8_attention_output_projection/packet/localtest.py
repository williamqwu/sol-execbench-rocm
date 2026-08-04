import torch, importlib, sys, json, time
sys.path.insert(0,'.')

import enum
if not hasattr(enum,'StrEnum'):
    class StrEnum(str, enum.Enum):
        def __str__(self): return self.value
    enum.StrEnum = StrEnum
import reference, kernel
dev='cuda:0'
torch.manual_seed(0)
K=N=7680

def check(M, atol, rtol, ratio=0.99):
    a=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
    w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)
    b=torch.randn(N,device=dev,dtype=torch.bfloat16)
    ref=reference.run(a,w,b)
    out=kernel.run(a,w,b)
    assert out.shape==ref.shape and out.dtype==ref.dtype,(out.shape,out.dtype)
    r=ref.float(); o=out.float()
    err=(o-r).abs()
    tol=atol+rtol*r.abs()
    matched=(err<=tol).float().mean().item()
    print(f"M={M:6d} matched={matched:.6f} maxabs={err.max().item():.5f} "
          f"maxrel={(err/(r.abs()+1e-9)).max().item():.4f} {'PASS' if matched>=ratio else 'FAIL'}")
    return matched>=ratio

Ms=[int(l['axes']['batch_size'])*int(l['axes']['seq_len']) for l in map(json.loads,open('workload.jsonl'))]
tols={}
for l in map(json.loads,open('workload.jsonl')):
    m=l['axes']['batch_size']*l['axes']['seq_len']; tols[m]=l['tolerance']
ok=True
for m in sorted(set(Ms)):
    ok &= check(m, tols[m]['max_atol'], tols[m]['max_rtol'])
print("ALL", ok)
