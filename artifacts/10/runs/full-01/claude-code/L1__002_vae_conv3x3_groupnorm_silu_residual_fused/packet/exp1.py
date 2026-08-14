import torch, torch.nn.functional as F, time

torch.manual_seed(0)
dev = 'cuda:0'

def ref(x, w1, n1w, n1b, w2, n2w, n2b, eps):
    num_groups = 32
    residual = x
    out = F.conv2d(x, w1, bias=None, stride=1, padding=1)
    out = F.group_norm(out, num_groups, weight=n1w, bias=n1b, eps=eps)
    out = F.silu(out)
    out = F.conv2d(out, w2, bias=None, stride=1, padding=1)
    out = F.group_norm(out, num_groups, weight=n2w, bias=n2b, eps=eps)
    out = F.silu(out)
    return out + residual

def mk(B,H,W,dtype=torch.float32):
    C=256
    x = torch.randn(B,C,H,W, device=dev, dtype=dtype)
    w1 = torch.randn(C,C,3,3, device=dev, dtype=dtype)
    n1w = torch.randn(C, device=dev, dtype=dtype)
    n1b = torch.randn(C, device=dev, dtype=dtype)
    w2 = torch.randn(C,C,3,3, device=dev, dtype=dtype)
    n2w = torch.randn(C, device=dev, dtype=dtype)
    n2b = torch.randn(C, device=dev, dtype=dtype)
    return x,w1,n1w,n1b,w2,n2w,n2b,1e-6

def stats(a, b, atol, rtol):
    d = (a.double()-b.double()).abs()
    thr = atol + rtol*b.double().abs()
    matched = (d<=thr).double().mean().item()
    return dict(maxabs=d.max().item(), matched=matched, refmax=b.abs().max().item(), refstd=b.std().item())

B,H,W = 2,64,64
args = mk(B,H,W)
o1 = ref(*args)
o2 = ref(*args)
print('determinism run-to-run:', stats(o1,o2,1.39e-7,1.19e-7))

# fp64 ground truth
args64 = [a.double() if torch.is_tensor(a) else a for a in args]
o64 = ref(*args64)
print('fp32 vs fp64 :', stats(o1, o64.float(), 1.39e-7, 1.19e-7))
print('out std', o1.std().item(), 'absmax', o1.abs().max().item())

# intermediate magnitudes
c1 = F.conv2d(args[0], args[1], padding=1)
print('conv1 out std', c1.std().item())

# channels_last variant of conv (different algorithm)
xcl = args[0].contiguous(memory_format=torch.channels_last)
w1cl = args[1].contiguous(memory_format=torch.channels_last)
c1cl = F.conv2d(xcl, w1cl, padding=1)
d = (c1.double()-c1cl.double()).abs()
print('conv1 nchw vs nhwc maxabs', d.max().item(), 'rel', (d/c1.double().abs().clamp(min=1e-6)).median().item())
