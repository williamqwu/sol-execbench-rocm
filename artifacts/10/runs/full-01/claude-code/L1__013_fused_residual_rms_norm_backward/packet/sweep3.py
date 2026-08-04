import torch, triton
import triton.language as tl
from variants import bench_ev, mk, DEV, HID, split, _bwd

# ---------------- reduce implementations over PART[P, N] ----------------------

@triton.jit
def _red_flat(PART, GW, P, N, BC: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BC + tl.arange(0, BC)
    m = cols < N
    acc = tl.zeros([BC], tl.float32)
    for i in range(P):
        acc += tl.load(PART + i * N + cols, mask=m, other=0.0)
    tl.store(GW + cols, acc, mask=m)


@triton.jit
def _red_2d(PART, TMP, P, N, SPLIT, BC: tl.constexpr):
    # grid (ceil(N/BC), SPLIT) -> TMP[SPLIT, N]
    cb = tl.program_id(0)
    sb = tl.program_id(1)
    cols = cb * BC + tl.arange(0, BC)
    m = cols < N
    acc = tl.zeros([BC], tl.float32)
    i = sb
    while i < P:
        acc += tl.load(PART + i * N + cols, mask=m, other=0.0)
        i += SPLIT
    tl.store(TMP + sb * N + cols, acc, mask=m)


def red_torch(part, N):
    return part.sum(0)


def red_flat(part, N, bc=128):
    gw = torch.empty(N, device=part.device, dtype=torch.float32)
    _red_flat[(triton.cdiv(N, bc),)](part, gw, part.shape[0], N, BC=bc, num_warps=4)
    return gw


def red_2stage(part, N, bc=128, split_=32):
    P = part.shape[0]
    tmp = torch.empty((split_, N), device=part.device, dtype=torch.float32)
    _red_2d[(triton.cdiv(N, bc), split_)](part, tmp, P, N, split_, BC=bc, num_warps=4)
    return tmp.sum(0)


if __name__ == "__main__":
    print("=== reduce over PART[P,2560] ===")
    for P in (256, 512, 1024, 2048):
        part = torch.randn(P, HID, device=DEV, dtype=torch.float32)
        ideal = P * HID * 4 / 1e9
        line = f"P={P:5d} ({P*HID*4/1e6:6.1f} MB)"
        t = bench_ev(lambda: red_torch(part, HID), (), iters=50)
        line += f" | torch {t*1000:7.1f}us"
        t = bench_ev(lambda: red_flat(part, HID), (), iters=50)
        line += f" | flat {t*1000:7.1f}us"
        for sp in (16, 64, 256):
            t = bench_ev(lambda s=sp: red_2stage(part, HID, split_=s), (), iters=50)
            line += f" | 2st{sp} {t*1000:6.1f}us"
        print(line)

    # correctness of reduces
    part = torch.randn(517, HID, device=DEV, dtype=torch.float32)
    r0 = red_torch(part, HID)
    for nm, f in (("flat", red_flat), ("2stage", red_2stage)):
        r = f(part, HID)
        print(nm, "maxdiff", (r - r0).abs().max().item())
