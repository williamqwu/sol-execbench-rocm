import sys, torch, triton, triton.language as tl
dev = 'cuda'
torch.manual_seed(0)
N = 1 << 22
x = torch.rand(N, device=dev) * 10
ref = torch.clamp(x / 448.0, min=1e-12)


@triton.jit
def k(X, O1, O2, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(X + i, mask=m)
    tl.store(O1 + i, tl.maximum(x / 448.0, 1e-12), mask=m)
    tl.store(O2 + i, tl.maximum(tl.fdiv(x, 448.0, ieee_rounding=True), 1e-12), mask=m)


o1 = torch.empty(N, device=dev)
o2 = torch.empty(N, device=dev)
k[(triton.cdiv(N, 1024),)](x, o1, o2, N, BLOCK=1024)
print("plain", (o1 == ref).float().mean().item())
print("ieee ", (o2 == ref).float().mean().item())
