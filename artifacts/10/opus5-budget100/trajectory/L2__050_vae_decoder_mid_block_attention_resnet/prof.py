import torch, time, sys
import torch.nn.functional as F

dev = 'cuda'
torch.manual_seed(0)

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1000

C=512
for (B,H,W) in [(1,32,32),(32,32,32),(2,64,64),(1,61,61)]:
    x = torch.randn(B,C,H,W,device=dev)
    w = torch.randn(C,C,3,3,device=dev)*0.01
    b = torch.randn(C,device=dev)
    gw = torch.randn(C,device=dev); gb=torch.randn(C,device=dev)
    t_conv = bench(lambda: F.conv2d(x,w,b,padding=1))
    xc = x.to(memory_format=torch.channels_last); wc = w.to(memory_format=torch.channels_last)
    t_convc = bench(lambda: F.conv2d(xc,wc,b,padding=1))
    t_gn = bench(lambda: F.group_norm(x,32,gw,gb,1e-6))
    S=H*W
    q = torch.randn(B,S,C,device=dev)
    t_qkv = bench(lambda: torch.matmul(q, torch.randn(C,C,device=dev).t()) if False else F.linear(q, w.view(C,-1)[:, :C].contiguous(), b))
    k = torch.randn(B,S,C,device=dev); v=torch.randn(B,S,C,device=dev)
    def attn():
        s = torch.matmul(q, k.transpose(-2,-1))*(C**-0.5)
        p = F.softmax(s,dim=-1)
        return torch.matmul(p,v)
    t_attn = bench(attn)
    def sdpa():
        return F.scaled_dot_product_attention(q.unsqueeze(1),k.unsqueeze(1),v.unsqueeze(1))
    try:
        t_sdpa = bench(sdpa)
    except Exception as e:
        t_sdpa = -1
    print(f"B{B} {H}x{W}: conv={t_conv:.3f} convCL={t_convc:.3f} gn={t_gn:.3f} lin={t_qkv:.3f} attn={t_attn:.3f} sdpa={t_sdpa:.3f}")
