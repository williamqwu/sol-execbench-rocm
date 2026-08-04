import torch, time, torch.nn.functional as F
torch.manual_seed(0)
dev='cuda'
B,C,H,W=32,512,32,32
x=torch.randn(B,C,H,W,device=dev)
w=torch.randn(C,C,3,3,device=dev)
b=torch.randn(C,device=dev)
g=torch.randn(C,device=dev)

def bench(f,n=20):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000

print("conv2d fp32 contig:", bench(lambda: F.conv2d(x,w,b,padding=1)))
xc=x.contiguous(memory_format=torch.channels_last); wc=w.contiguous(memory_format=torch.channels_last)
print("conv2d fp32 chlast:", bench(lambda: F.conv2d(xc,wc,b,padding=1)))
print("groupnorm:", bench(lambda: F.group_norm(x,32,g,b,1e-5)))
print("groupnorm chlast:", bench(lambda: F.group_norm(xc,32,g,b,1e-5)))
print("silu:", bench(lambda: F.silu(x)))
s=H*W
h=torch.randn(B,s,C,device=dev)
wl=torch.randn(C,C,device=dev)
print("linear:", bench(lambda: F.linear(h,wl,b)))
q=torch.randn(B,1,s,C,device=dev)
print("attn:", bench(lambda: torch.matmul(torch.softmax(torch.matmul(q,q.transpose(-2,-1))*0.04,dim=-1),q)))
print("sdpa:", bench(lambda: F.scaled_dot_product_attention(q,q,q)))
