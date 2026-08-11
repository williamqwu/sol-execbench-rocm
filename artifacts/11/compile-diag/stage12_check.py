"""Confirm: Inductor's fused silu*up keeps the intermediate in fp32 (one rounding),
eager rounds silu to bf16 first (two roundings)."""
import torch, torch.nn.functional as F
dev = "cuda:0"
torch.manual_seed(1)
gate = (torch.randn(1024, 2048, device=dev) * 3).to(torch.bfloat16)
up   = (torch.randn(1024, 2048, device=dev) * 3).to(torch.bfloat16)

def f(g, u):
    return F.silu(g) * u
fc = torch.compile(f, dynamic=False)

e = f(gate, up)                                              # eager
c = fc(gate, up); torch.cuda.synchronize()                   # compiled
two_round = (F.silu(gate) * up)                              # bf16 silu then bf16 mul
one_round = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
g64 = F.silu(gate.double()) * up.double()

print("compiled == eager                :", bool(torch.equal(c, e)))
print("eager    == bf16-silu-then-mul   :", bool(torch.equal(e, two_round)))
print("compiled == fp32-intermediate    :", bool(torch.equal(c, one_round)))
print("n elements differing eager/compiled:", int((c != e).sum()), "of", e.numel())
for tag, t in (("eager", e), ("compiled", c)):
    d = (t.double() - g64).abs()
    print(f"{tag:9s} vs f64: max_abs={float(d.max()):.6e} mean_abs={float(d.mean()):.6e}")

# how much of compiled-vs-eager is the elided bf16 rounding, and how much is
# tl.sigmoid differing from aten sigmoid?
one = one_round.double()
for tag, t in (("fp32-intermediate (hand)", one),):
    d = (t - g64).abs()
    print(f"{tag:24s} vs f64: max_abs={float(d.max()):.6e} mean_abs={float(d.mean()):.6e}")
print("n elements compiled != fp32-intermediate:", int((c != one_round).sum()), "of", c.numel())
print("n elements eager    != fp32-intermediate:", int((e != one_round).sum()), "of", c.numel())
sg_c = torch.compile(lambda x: torch.sigmoid(x), dynamic=False)
xs = gate.float()
print("tl.sigmoid == aten sigmoid (fp32):", bool(torch.equal(sg_c(xs), torch.sigmoid(xs))),
      " max diff:", float((sg_c(xs).double() - torch.sigmoid(xs).double()).abs().max()))
