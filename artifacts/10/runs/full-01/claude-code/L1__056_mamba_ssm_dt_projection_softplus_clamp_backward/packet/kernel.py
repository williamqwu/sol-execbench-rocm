import torch
import triton
import triton.language as tl


# num_heads is a constant (40) in this definition, but keep the kernels generic
# over it via NH / HB / VEC:  H == NH * VEC, HB = next_pow2(NH).
@triton.jit
def _dt_bwd_kernel(GO, DTB, ACT, OUT, PART, R, tmin, tmax,
                   H: tl.constexpr, ROWS: tl.constexpr, HB: tl.constexpr,
                   VEC: tl.constexpr, NH: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * ROWS + tl.arange(0, ROWS)
    hb = tl.arange(0, HB)
    v = tl.arange(0, VEC)

    off = r[:, None, None] * H + hb[None, :, None] * VEC + v[None, None, :]
    m = (r[:, None, None] < R) & (hb[None, :, None] < NH)

    go = tl.load(GO + off, mask=m, other=0.0).to(tl.float32)
    act = tl.load(ACT + off, mask=m, other=0.0).to(tl.float32)
    dtb = tl.load(DTB + off, mask=m, other=0.0).to(tl.float32)

    keep = (act > tmin) & (act < tmax)
    g = tl.where(keep, go, 0.0) * (1.0 / (1.0 + tl.exp(-dtb)))

    tl.store(OUT + off, g.to(tl.bfloat16), mask=m)

    p = tl.sum(g, axis=0)
    poff = pid * (HB * VEC) + hb[:, None] * VEC + v[None, :]
    tl.store(PART + poff, p)


@triton.jit
def _dt_bwd_reduce(PART, BIAS, G, BG: tl.constexpr, HB: tl.constexpr,
                   VEC: tl.constexpr, NH: tl.constexpr):
    hb = tl.arange(0, HB)
    v = tl.arange(0, VEC)
    acc = tl.zeros((HB, VEC), dtype=tl.float32)
    for i in range(0, G, BG):
        rows = i + tl.arange(0, BG)
        off = rows[:, None, None] * (HB * VEC) + hb[None, :, None] * VEC + v[None, None, :]
        x = tl.load(PART + off, mask=rows[:, None, None] < G, other=0.0)
        acc += tl.sum(x, axis=0)
    tl.store(BIAS + hb[:, None] * VEC + v[None, :], acc.to(tl.bfloat16),
             mask=hb[:, None] < NH)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def _pick(R):
    if R <= 1024:
        return 2, 1
    if R <= 8192:
        return 8, 2
    if R <= 65536:
        return 32, 4
    return 128, 4


def run(grad_output, dt_with_bias, dt_activated, time_step_min, time_step_max):
    B, S, H = grad_output.shape
    R = B * S

    # split H into VEC-wide contiguous chunks so loads stay vectorised
    VEC = 4 if H % 4 == 0 else (2 if H % 2 == 0 else 1)
    NH = H // VEC
    HB = _next_pow2(NH)

    ROWS, nw = _pick(R)
    G = (R + ROWS - 1) // ROWS

    out = torch.empty_like(grad_output)
    part = torch.empty((G, HB * VEC), dtype=torch.float32, device=grad_output.device)
    bias = torch.empty((H,), dtype=torch.bfloat16, device=grad_output.device)

    _dt_bwd_kernel[(G,)](
        grad_output, dt_with_bias, dt_activated, out, part, R,
        time_step_min, time_step_max,
        H=H, ROWS=ROWS, HB=HB, VEC=VEC, NH=NH,
        num_warps=nw, num_stages=1,
    )

    BG = min(_next_pow2(G), 256)
    _dt_bwd_reduce[(1,)](part, bias, G, BG=BG, HB=HB, VEC=VEC, NH=NH,
                         num_warps=4, num_stages=1)

    return out, bias
