import torch, torch.nn.functional as F, time
dev='cuda:0'; C=256
torch.manual_seed(0)

def bench(fn,n=20,w=5):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3

for (B,H,W) in [(2,64,64),(4,128,128),(32,128,128),(1,1024,1024),(2,256,256)]:
    x=torch.randn(B,C,H,W,device=dev); w1=torch.randn(C,C,3,3,device=dev)
    xcl=x.contiguous(memory_format=torch.channels_last)
    w1cl=w1.contiguous(memory_format=torch.channels_last)
    t_nchw=bench(lambda: F.conv2d(x,w1,None,1,1),10,5)
    t_nhwc=bench(lambda: F.conv2d(xcl,w1cl,None,1,1),10,5)
    t_tr = bench(lambda: x.contiguous(memory_format=torch.channels_last),10,5)
    o_a=F.conv2d(x,w1,None,1,1); o_b=F.conv2d(xcl,w1cl,None,1,1)
    print(f'B{B} {H}x{W}: NCHW={t_nchw:.3f}ms NHWC={t_nhwc:.3f}ms transpose={t_tr:.3f}ms  bitexact={torch.equal(o_a,o_b)} out_is_cl={o_b.is_contiguous(memory_format=torch.channels_last)}')
