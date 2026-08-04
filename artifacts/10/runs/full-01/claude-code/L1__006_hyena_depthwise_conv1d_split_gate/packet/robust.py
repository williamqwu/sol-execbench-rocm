"""Numerics robustness across input distributions/seeds + Python-overhead micro-bench."""
import time
import torch
import torch.nn.functional as F
import kernel as K

D = 256
ATOL_MIN, RTOL = 1.97e-7, 1.1920928955078125e-07
RATIO = 0.99


def gen(kind, B, S, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    def r(shape):
        if kind == "normal":
            return torch.randn(shape, device="cuda", generator=g)
        if kind == "uniform":
            return torch.rand(shape, device="cuda", generator=g)
        if kind == "uniform_pm1":
            return torch.rand(shape, device="cuda", generator=g) * 2 - 1
        if kind == "big":
            return torch.randn(shape, device="cuda", generator=g) * 100
        if kind == "small":
            return torch.randn(shape, device="cuda", generator=g) * 1e-3
        raise ValueError(kind)
    return r((B, 768, S)), r((768, 1, 3)), r((768,))


def ratios(ref, got):
    out = []
    for r, m in zip(ref, got):
        d = (r.double() - m.double()).abs()
        thr = ATOL_MIN + RTOL * r.double().abs()
        out.append((d <= thr).double().mean().item())
    return out


print("=== numerics across distributions x seeds (need every ratio >= 0.99) ===")
worst_overall = 1.0
for kind in ["normal", "uniform", "uniform_pm1", "big", "small"]:
    worst = 1.0
    for seed in range(4):
        for (B, S) in [(1, 512), (4, 4096), (2, 293), (64, 128), (32, 512)]:
            u, w, bi = gen(kind, B, S, seed)
            ref = F.conv1d(u, w, bias=bi, padding=2, groups=768)[..., :S]
            r0 = ref[:, :D, :]; r1 = ref[:, D:2*D, :]; rv = ref[:, 2*D:, :]
            expect = (rv * r0, r0, r1)
            got = K.run(u, w, bi)
            worst = min(worst, min(ratios(expect, got)))
    worst_overall = min(worst_overall, worst)
    print(f"  {kind:12s} worst matched ratio = {worst:.5f}  {'PASS' if worst >= RATIO else 'FAIL'}")
print(f"  OVERALL worst = {worst_overall:.5f}")

# ---------------- python overhead ----------------
print()
print("=== python-side overhead options ===")
B, S = 1, 512
def cpu(fn, iters=4000):
    for _ in range(100): fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters): fn()
    dt = (time.perf_counter()-t)/iters*1e6
    torch.cuda.synchronize()
    return dt

big = torch.empty((3, B, D, S), device="cuda")
print(f"  {'empty((3,B,D,S))':38s} {cpu(lambda: torch.empty((3,B,D,S),device='cuda')):7.2f}us")
print(f"  {'3x empty((B,D,S))':38s} {cpu(lambda: [torch.empty((B,D,S),device='cuda') for _ in range(3)]):7.2f}us")
print(f"  {'big[0],big[1],big[2]':38s} {cpu(lambda: (big[0],big[1],big[2])):7.2f}us")
print(f"  {'big.unbind(0)':38s} {cpu(lambda: big.unbind(0)):7.2f}us")
print(f"  {'torch.unbind(big,0)':38s} {cpu(lambda: torch.unbind(big,0)):7.2f}us")
flat = torch.empty(3*B*D*S, device="cuda")
n = B*D*S
print(f"  {'3x flat.view narrow':38s} {cpu(lambda: (flat[:n].view(B,D,S),flat[n:2*n].view(B,D,S),flat[2*n:].view(B,D,S))):7.2f}us")
print(f"  {'full run()':38s} {cpu(lambda: K.run(*gen0)):7.2f}us" if False else "")
u, w, bi = gen("normal", B, S, 0)
print(f"  {'full run()':38s} {cpu(lambda: K.run(u,w,bi)):7.2f}us")
