import torch, triton, triton.language as tl
dev = 'cuda'
torch.manual_seed(0)
N = 1 << 22
x = torch.rand(N, device=dev) * 10
r_div = x / 448.0
r_max = torch.maximum(x / 448.0, torch.tensor(1e-12, device=dev))
c = torch.full((N,), 448.0, device=dev)
r_tdiv = x / c


@triton.jit
def k(X, C, O1, O2, O3, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(X + i, mask=m)
    c = tl.load(C + i, mask=m)
    tl.store(O1 + i, x / 448.0, mask=m)
    tl.store(O2 + i, x / c, mask=m)
    tl.store(O3 + i, x * (1.0 / 448.0), mask=m)


o = [torch.empty(N, device=dev) for _ in range(3)]
k[(triton.cdiv(N, 1024),)](x, c, *o, N, BLOCK=1024)
for nm, oo in zip(["const/", "tensor/", "*recip"], o):
    print(nm, "vs torch x/448:", (oo == r_div).float().mean().item(),
          " vs torch x/c:", (oo == r_tdiv).float().mean().item())
print("torch x/448 vs x/c:", (r_div == r_tdiv).float().mean().item())
print("torch max vs div:", (r_max == r_div).float().mean().item())
