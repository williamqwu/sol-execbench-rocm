import torch, torch.nn.functional as F, time

dev='cuda:0'
torch.manual_seed(0)

def bench(fn, n=20, w=5):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t)/n*1e3  # ms

C=256
def mk(B,H,W):
    return (torch.randn(B,C,H,W,device=dev), torch.randn(C,C,3,3,device=dev),
            torch.randn(C,device=dev), torch.randn(C,device=dev),
            torch.randn(C,C,3,3,device=dev), torch.randn(C,device=dev), torch.randn(C,device=dev), 1e-6)

def ref(x,w1,n1w,n1b,w2,n2w,n2b,eps):
    r=x
    o=F.conv2d(x,w1,None,1,1); o=F.group_norm(o,32,n1w,n1b,eps); o=F.silu(o)
    o=F.conv2d(o,w2,None,1,1); o=F.group_norm(o,32,n2w,n2b,eps); o=F.silu(o)
    return o+r

print('tf32 allowed:', torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
print('fp32 precision:', torch.get_float32_matmul_precision())

for (B,H,W) in [(2,64,64),(4,128,128),(1,1024,1024),(32,128,128)]:
    a=mk(B,H,W)
    N = B*C*H*W
    flops = 2*2*N*C*9
    bytes_ = N*4
    t_full = bench(lambda: ref(*a), n=10, w=5)
    x=a[0]; w1=a[1]
    t_conv = bench(lambda: F.conv2d(x,w1,None,1,1), n=10, w=5)
    c1 = F.conv2d(x,w1,None,1,1)
    t_gn = bench(lambda: F.group_norm(c1,32,a[2],a[3],1e-6), n=10, w=5)
    t_silu = bench(lambda: F.silu(c1), n=10, w=5)
    print(f'B={B} H={H} W={W} N={N/1e6:.1f}M | full={t_full:.3f}ms conv={t_conv:.3f} gn={t_gn:.3f} silu={t_silu:.3f} | 2conv+rest={2*t_conv+2*t_gn+2*t_silu:.3f}')
    print(f'    conv TFLOPS={flops/2/(t_conv/1e3)/1e12:.1f}  full-eff-TFLOPS={flops/(t_full/1e3)/1e12:.1f}  memBW_gn={3*bytes_/(t_gn/1e3)/1e12:.2f} TB/s  memBW_silu={2*bytes_/(t_silu/1e3)/1e12:.2f} TB/s')
