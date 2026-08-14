"""Identify the reference conv1d's exact float32 rounding order (bit-exact match)."""
import torch
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True

B, S, C = 2, 1024, 768
g = torch.Generator(device="cuda").manual_seed(0)
u = torch.randn((B, C, S), device="cuda", generator=g)
w = torch.randn((C, 1, 3), device="cuda", generator=g)
bi = torch.randn((C,), device="cuda", generator=g)

ref = F.conv1d(u, w, bias=bi, padding=2, groups=C)[..., :S]

# shifted inputs: a0 = u[t], a1 = u[t-1], a2 = u[t-2]
a0 = u
a1 = torch.zeros_like(u); a1[:, :, 1:] = u[:, :, :-1]
a2 = torch.zeros_like(u); a2[:, :, 2:] = u[:, :, :-2]

w0 = w[:, 0, 0].view(1, C, 1)
w1 = w[:, 0, 1].view(1, C, 1)
w2 = w[:, 0, 2].view(1, C, 1)
bb = bi.view(1, C, 1)


def fma(x, y, z):
    return (x.double() * y.double() + z.double()).float()


exact = (a2.double() * w0.double() + a1.double() * w1.double()
         + a0.double() * w2.double() + bb.double())

cands = {
    # plain-multiply then add, various orders
    "A: (w0a2 + w1a1) + w2a0 + b": ((w0 * a2 + w1 * a1) + w2 * a0) + bb,
    "B: b + w0a2 + w1a1 + w2a0": ((bb + w0 * a2) + w1 * a1) + w2 * a0,
    "C: (w2a0 + w1a1) + w0a2 + b": ((w2 * a0 + w1 * a1) + w0 * a2) + bb,
    "D: b + w2a0 + w1a1 + w0a2": ((bb + w2 * a0) + w1 * a1) + w0 * a2,
    "E: w0a2 + (w1a1 + w2a0) + b": (w0 * a2 + (w1 * a1 + w2 * a0)) + bb,
    # fma chains, accumulator starts at bias
    "F: fma(w2,a0,fma(w1,a1,fma(w0,a2,b)))": fma(w2, a0, fma(w1, a1, fma(w0, a2, bb))),
    "G: fma(w0,a2,fma(w1,a1,fma(w2,a0,b)))": fma(w0, a2, fma(w1, a1, fma(w2, a0, bb))),
    # fma chain from 0, bias added last
    "H: fma(w2,a0,fma(w1,a1,w0*a2)) + b": fma(w2, a0, fma(w1, a1, w0 * a2)) + bb,
    "I: fma(w0,a2,fma(w1,a1,w2*a0)) + b": fma(w0, a2, fma(w1, a1, w2 * a0)) + bb,
    # correctly-rounded (exact) result
    "X: exact fp64 -> fp32": exact.float(),
}

print(f"{'variant':45s} {'bitexact%':>10s} {'ulp<=0.5%':>10s}")
for k, v in cands.items():
    be = (v == ref).double().mean().item() * 100
    d = (v.double() - ref.double()).abs()
    thr = 2e-7 + 1.192e-7 * ref.double().abs()
    within = (d <= thr).double().mean().item() * 100
    print(f"{k:45s} {be:10.4f} {within:10.4f}")

# also: how does the *reference itself* vary run to run?
ref2 = F.conv1d(u, w, bias=bi, padding=2, groups=C)[..., :S]
print("ref determinism (bitexact%):", (ref2 == ref).double().mean().item() * 100)
