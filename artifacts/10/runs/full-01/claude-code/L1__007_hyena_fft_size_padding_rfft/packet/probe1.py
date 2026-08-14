import torch, triton, triton.language as tl, time
dev = 'cuda:0'


@triton.jit
def k_div(SRC, RE, IM, M, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid*BLOCK + tl.arange(0, BLOCK); m = offs < M
    two = tl.arange(0, 2)
    v = tl.load(SRC + offs[:, None]*2 + two[None, :], mask=m[:, None], other=0.)
    re, im = tl.split(v)
    tl.store(RE + offs, re/N, mask=m); tl.store(IM + offs, im/N, mask=m)


@triton.jit
def k_mul(SRC, RE, IM, M, INV, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid*BLOCK + tl.arange(0, BLOCK); m = offs < M
    two = tl.arange(0, 2)
    v = tl.load(SRC + offs[:, None]*2 + two[None, :], mask=m[:, None], other=0.)
    re, im = tl.split(v)
    tl.store(RE + offs, re*INV, mask=m); tl.store(IM + offs, im*INV, mask=m)


for S in [1024, 1423, 211]:
    B = 4
    x = torch.randn(B, 256, S, device=dev, dtype=torch.float32)
    n = 2*S
    z = torch.fft.rfft(x, n=n)
    gr = (z/n).real.contiguous(); gi = (z/n).imag.contiguous()
    flat = z.view(torch.float32).reshape(-1); M = flat.numel()//2
    for name, kern, arg in [("div", k_div, float(n)), ("mul", k_mul, 1.0/n)]:
        re = torch.empty(M, device=dev); im = torch.empty(M, device=dev)
        kern[(triton.cdiv(M, 1024),)](flat, re, im, M, arg, BLOCK=1024)
        ex = torch.equal(re.reshape(gr.shape), gr) and torch.equal(im.reshape(gi.shape), gi)
        err = max((re.reshape(gr.shape)-gr).abs().max().item(), (im.reshape(gi.shape)-gi).abs().max().item())
        print(f"S={S} {name}: bitexact={ex} maxerr={err:.3e}")
    zf = torch.fft.rfft(x, n=n, norm='forward')
    print(f"   norm=forward bitexact: {torch.equal(torch.view_as_real(zf), torch.view_as_real(z/n))}")
