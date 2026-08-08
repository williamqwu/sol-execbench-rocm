import sys, torch, triton, triton.language as tl
sys.path.insert(0, "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear")
dev = 'cuda'
torch.manual_seed(0)
E = tl.constexpr(448.0)

N = 1 << 22
x = (torch.randn(N, device=dev, dtype=torch.bfloat16)).float()
s = torch.rand(N, device=dev) * 0.01 + 1e-3
ref = torch.clamp(x / s, -448.0, 448.0).to(torch.float8_e4m3fn)


@triton.jit
def k(X, S, O1, O2, O3, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(X + i, mask=m)
    s = tl.load(S + i, mask=m)
    a = x / s
    b = tl.fdiv(x, s, ieee_rounding=True)
    c = x * (1.0 / s)
    tl.store(O1 + i, tl.minimum(tl.maximum(a, -448.0), 448.0).to(tl.float8e4nv), mask=m)
    tl.store(O2 + i, tl.minimum(tl.maximum(b, -448.0), 448.0).to(tl.float8e4nv), mask=m)
    tl.store(O3 + i, tl.minimum(tl.maximum(c, -448.0), 448.0).to(tl.float8e4nv), mask=m)


o = [torch.empty(N, dtype=torch.float8_e4m3fn, device=dev) for _ in range(3)]
k[(triton.cdiv(N, 1024),)](x, s, *o, N, BLOCK=1024)
for nm, oo in zip(["plain /", "ieee fdiv", "mul recip"], o):
    print(nm, (oo.view(torch.uint8) == ref.view(torch.uint8)).float().mean().item())
