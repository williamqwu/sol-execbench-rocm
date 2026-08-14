"""(a) per-output matched ratios for candidate precisions, (b) bit-exact order search."""
import itertools, json
import torch
import torch.nn.functional as F

B, S, C = 2, 1024, 768
D = 256
g = torch.Generator(device="cuda").manual_seed(0)
u = torch.randn((B, C, S), device="cuda", generator=g)
w = torch.randn((C, 1, 3), device="cuda", generator=g)
bi = torch.randn((C,), device="cuda", generator=g)

uc_ref = F.conv1d(u, w, bias=bi, padding=2, groups=C)[..., :S]
x0r = uc_ref[:, :D, :]; x1r = uc_ref[:, D:2 * D, :]; vr = uc_ref[:, 2 * D:3 * D, :]
vgr = vr * x0r

a0 = u
a1 = torch.zeros_like(u); a1[:, :, 1:] = u[:, :, :-1]
a2 = torch.zeros_like(u); a2[:, :, 2:] = u[:, :, :-2]
W = [w[:, 0, i].view(1, C, 1) for i in range(3)]
A = [a2, a1, a0]          # term i pairs W[i] with A[i]
bb = bi.view(1, C, 1)

ATOL, RTOL, RATIO = 1.98e-7, 1.1920928955078125e-07, 0.99


def ratios(uc, vg):
    out = []
    for r, m in ((x0r, uc[:, :D, :]), (x1r, uc[:, D:2 * D, :]), (vgr, vg)):
        d = (r.double() - m.double()).abs()
        thr = ATOL + RTOL * r.double().abs()
        out.append((d <= thr).double().mean().item())
    return out


print("=== (a) precision candidates: matched ratio per output (need >=0.99 each) ===")

# fp32 naive (what kernel.py does now)
uc32 = (W[0] * A[0] + W[1] * A[1]) + W[2] * A[2] + bb
# fp64 exact
uc64 = (W[0].double() * A[0].double() + W[1].double() * A[1].double()
        + W[2].double() * A[2].double() + bb.double())

cands = {
    "fp32 naive, vg=fp32(v*x0)": (uc32, uc32[:, 2 * D:, :] * uc32[:, :D, :]),
    "fp64 conv -> fp32, vg=fp32(v*x0)": (uc64.float(), uc64[:, 2 * D:, :].float() * uc64[:, :D, :].float()),
    "fp64 conv, vg=fp64(v*x0)->fp32": (uc64.float(), (uc64[:, 2 * D:, :] * uc64[:, :D, :]).float()),
}
for k, (uc, vg) in cands.items():
    r = ratios(uc.float() if uc.dtype == torch.float64 else uc,
               vg.float() if vg.dtype == torch.float64 else vg)
    print(f"  {k:42s} x0={r[0]:.5f} x1={r[1]:.5f} vgated={r[2]:.5f}  "
          f"{'PASS' if min(r) >= RATIO else 'FAIL'}")

print()
print("=== (b) bit-exact accumulation-order search vs reference conv ===")


def fma(x, y, z):
    return torch.addcmul(z.double(), x.double(), y.double()).float()


best = []
for perm in itertools.permutations(range(3)):
    for use_fma in (False, True):
        for bias_pos in ("first", "last"):
            if use_fma:
                acc = bb if bias_pos == "first" else W[perm[0]] * A[perm[0]]
                start = 0 if bias_pos == "first" else 1
                for i in perm[start:]:
                    acc = fma(W[i], A[i], acc)
                if bias_pos == "last":
                    acc = acc + bb
            else:
                acc = bb if bias_pos == "first" else W[perm[0]] * A[perm[0]]
                start = 0 if bias_pos == "first" else 1
                for i in perm[start:]:
                    acc = acc + W[i] * A[i]
                if bias_pos == "last":
                    acc = acc + bb
            be = (acc == uc_ref).double().mean().item() * 100
            best.append((be, f"perm={perm} fma={int(use_fma)} bias={bias_pos}"))

best.sort(reverse=True)
for be, name in best[:6]:
    print(f"  {be:8.4f}%  {name}")
print(f"  ({(uc64.float() == uc_ref).double().mean().item()*100:8.4f}%  fp64-exact)")
