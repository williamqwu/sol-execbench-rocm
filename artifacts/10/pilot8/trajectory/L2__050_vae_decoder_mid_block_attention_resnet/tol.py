import torch, math, sys
import reference, check
dev='cuda'
import torch.nn.functional as F
for (B,H,W,atol) in check.WL:
    args=check.mkargs(B,H,W)
    y=reference.run(*args)
    ya=y.abs()
    print(f"B{B} {H}x{W} atol={atol:.2e}: |y| med={ya.median().item():.3f} p99={ya.flatten().kthvalue(int(ya.numel()*0.99))[0].item():.2f} max={ya.max().item():.1f}")
