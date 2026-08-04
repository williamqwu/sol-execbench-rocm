import torch, time, torch.nn.functional as F, check
dev='cuda'
def bench(f,n=20):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000
for (B,H,W) in [(32,32,32),(1,32,32),(2,64,64),(4,16,16),(1,16,16)]:
    a=check.mkargs(B,H,W); C=512; S=H*W
    x=a[0]; w=a[4]; g=a[2]; bi=a[3]
    tconv=bench(lambda: F.conv2d(x,w,a[5],padding=1))
    tgn=bench(lambda: F.group_norm(x,32,g,bi,1e-5))
    tsilu=bench(lambda: F.silu(x))
    h=torch.randn(B,S,C,device=dev)
    tlin=bench(lambda: F.linear(h,a[14],a[15]))
    q=h.view(B,1,S,C)
    tattn=bench(lambda: torch.matmul(torch.softmax(torch.matmul(q,q.transpose(-2,-1))*0.044,dim=-1),q))
    tot=bench(lambda: __import__('reference').run(*a))
    print(f"B{B} {H}x{W}: total={tot:.3f} 4conv={4*tconv:.3f} 4gn={4*tgn:.3f} 4silu={4*tsilu:.3f} 4lin={4*tlin:.3f} attn={tattn:.3f} sum={4*tconv+4*tgn+4*tsilu+4*tlin+tattn:.3f}")
