import torch, triton, triton.language as tl
import torch.nn.functional as F

@triton.jit
def mom(X, MU, RS, N, eps, NT: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)
    base = i*N
    lane = tl.arange(0, NT)
    mean = tl.zeros([NT], tl.float32)
    m2 = tl.zeros([NT], tl.float32)
    nf = tl.zeros([NT], tl.float32)
    # ATen strided order: thread t handles j = t, t+NT, t+2NT, ...
    for off in tl.range(0, N, NT):
        o = off + lane
        m = o < N
        x = tl.load(X + base + o, mask=m, other=0.0)
        d = x - mean
        nf2 = nf + 1.0
        nm = mean + d / nf2
        nd = x - nm
        mean = tl.where(m, nm, mean)
        m2 = tl.where(m, m2 + d * nd, m2)
        nf = tl.where(m, nf2, nf)
    # tree combine, ATen warp_reduce order: offsets NT/2 .. 1
    o = NT // 2
    while o > 0:
        b_mean = tl.sum(tl.where(lane[:, None] == (lane[None, :] - o), mean[None, :], 0.0), 1)
        b_m2   = tl.sum(tl.where(lane[:, None] == (lane[None, :] - o), m2[None, :], 0.0), 1)
        b_nf   = tl.sum(tl.where(lane[:, None] == (lane[None, :] - o), nf[None, :], 0.0), 1)
        d = b_mean - mean
        nn = nf + b_nf
        nb = b_nf / nn
        cm = mean + d * nb
        cm2 = m2 + b_m2 + d * d * nf * nb
        take = (nf != 0.0) & (b_nf != 0.0)
        mean = tl.where(take, cm, tl.where(nf == 0.0, b_mean, mean))
        m2 = tl.where(take, cm2, tl.where(nf == 0.0, b_m2, m2))
        nf = tl.where(take, nn, tl.where(nf == 0.0, b_nf, nf))
        o = o // 2
    mu = tl.sum(tl.where(lane == 0, mean, 0.0))
    v = tl.sum(tl.where(lane == 0, m2, 0.0))
    n_ = tl.sum(tl.where(lane == 0, nf, 0.0))
    tl.store(MU + i, mu)
    tl.store(RS + i, tl.rsqrt(v / n_ + eps))

torch.manual_seed(0)
for B,H,W in [(1,32,32),(16,32,32),(2,41,41),(1,16,16),(2,64,64)]:
    C=512; ng=32; D=C//ng; HxW=H*W; N=D*HxW
    x=torch.randn(B,C,H,W,device='cuda')
    g=torch.ones(C,device='cuda'); bb=torch.zeros(C,device='cuda')
    # ATen reference moments
    xr = x.view(B*ng, N).double()
    NT = 64 if N < 512 else 512
    mu=torch.empty(B*ng,device='cuda'); rs=torch.empty(B*ng,device='cuda')
    mom[(B*ng,)](x, mu, rs, N, 1e-6, NT, num_warps=NT//64)
    _o, amu, ars = torch.native_group_norm(x, g, bb, B, C, HxW, ng, 1e-6)
    print(f"B{B} {H}x{W} NT={NT}: mean mism={(mu!=amu.flatten()).sum().item()}/{mu.numel()}  rstd mism={(rs!=ars.flatten()).sum().item()}")
