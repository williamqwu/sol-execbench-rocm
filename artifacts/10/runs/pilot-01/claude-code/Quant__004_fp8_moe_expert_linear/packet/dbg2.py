import torch, triton, triton.language as tl

@triton.jit
def k(X, O, N, MODE: tl.constexpr):
    i = tl.arange(0, 1024)
    x = tl.load(X + i, mask=i < N, other=0.)
    if MODE == 0:
        o = x / 448.0
    elif MODE == 1:
        o = tl.fdiv(x, 448.0, ieee_rounding=True)
    elif MODE == 2:
        o = tl.fdiv(x, tl.full((1024,), 448.0, tl.float32), ieee_rounding=True)
    else:
        o = tl.math.fma(x, 0.0, x / 448.0)
    tl.store(O + i, o, mask=i < N)

x = torch.randn(1024, device='cuda') * 100
ref_o = x / 448.0
for m in (0, 1, 2):
    o = torch.empty_like(x)
    k[(1,)](x, o, 1024, MODE=m)
    print("div-const mode", m, "bitexact", torch.equal(o, ref_o),
          "maxdiff", (o - ref_o).abs().max().item())

# amax reduction then divide (as in kernel)
@triton.jit
def k2(X, S, M, N: tl.constexpr, MODE: tl.constexpr):
    r = tl.arange(0, 128)
    m = tl.program_id(0)
    x = tl.load(X + m * N + r).to(tl.float32)
    a = tl.max(tl.abs(x), axis=0)
    if MODE == 0:
        s = a / 448.0
    else:
        s = tl.fdiv(a, 448.0, ieee_rounding=True)
    tl.store(S + m, tl.maximum(s, 1e-12))

xb = (torch.randn(64, 128, device='cuda', dtype=torch.bfloat16)).float()
sref = torch.clamp(xb.abs().amax(dim=1) / 448.0, min=1e-12)
for m in (0, 1):
    s = torch.empty(64, device='cuda')
    k2[(64,)](xb, s, 64, N=128, MODE=m)
    print("scale mode", m, "bitexact", torch.equal(s, sref), "maxdiff", (s - sref).abs().max().item())
