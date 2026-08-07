import torch
import triton
import triton.language as tl

# Flux multi-axis RoPE: axes_dim = [16, 56, 56], total_dim = 128.
# Concatenated half-dims: 8 + 28 + 28 = 64 frequency bands.
#
# Output column c in [0,128) maps to:
#   band index      b = c // 2                  (indexes the 64-long concat freq table)
#   axis            0 if c < 16, 1 if c < 72, else 2
# because the per-axis half-dims (8, 28, 28) tile [0,64) exactly in the same
# order the reference concatenates them.

TOTAL_DIM = 128
N_BANDS = 64
_TOTAL_DIM = tl.constexpr(128)
_N_BANDS = tl.constexpr(64)


_EXPS_CACHE = {}


def _exponents(dev):
    """The 64 concatenated frequency exponents: arange(h)/h for h in (8,28,28).

    Built with torch so the division rounds exactly as the reference's does;
    Triton's f32 divide differs by ~1ulp here, which is enough to blow the
    tolerance once multiplied by positions of order seq_len.

    This is a structural constant of the problem (it does not depend on the
    inputs, only on the fixed axes_dim), so it is cached per device. The
    input-dependent work -- pow, cos, sin -- is all still done per call.
    """
    e = _EXPS_CACHE.get(dev)
    if e is None:
        a8 = torch.arange(8, dtype=torch.float32, device=dev) / 8
        a28 = torch.arange(28, dtype=torch.float32, device=dev) / 28
        e = torch.cat([a8, a28, a28])
        _EXPS_CACHE[dev] = e
    return e


@triton.jit
def _rope_kernel(IDS, POW, COS, SIN, S, BLOCK_S: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    m = rows < S

    c = tl.arange(0, _TOTAL_DIM)
    b = c // 2

    # freq_bands = 1.0 / (theta ** (arange(h)/h)); POW holds theta**exp.
    # div_rn matches torch's IEEE round-to-nearest reciprocal bit-for-bit.
    fb = tl.math.div_rn(1.0, tl.load(POW + b))

    p0 = tl.load(IDS + rows * 3 + 0, mask=m, other=0.0)
    p1 = tl.load(IDS + rows * 3 + 1, mask=m, other=0.0)
    p2 = tl.load(IDS + rows * 3 + 2, mask=m, other=0.0)

    pos = tl.where(
        c[None, :] < 16,
        p0[:, None],
        tl.where(c[None, :] < 72, p1[:, None], p2[:, None]),
    )

    ang = pos * fb[None, :]

    off = rows[:, None] * _TOTAL_DIM + c[None, :]
    mm = m[:, None]
    tl.store(COS + off, tl.cos(ang), mask=mm)
    tl.store(SIN + off, tl.sin(ang), mask=mm)


@torch.no_grad()
def run(ids: torch.Tensor, theta: float):
    ids = ids.contiguous()
    S = ids.shape[0]
    dev = ids.device

    # Python-float base: same op the reference uses, and bit-identical to it.
    # (Wrapping theta in a device tensor instead costs a ~17us H2D copy.)
    powv = theta ** _exponents(dev)

    freqs_cos = torch.empty((S, TOTAL_DIM), dtype=torch.float32, device=dev)
    freqs_sin = torch.empty((S, TOTAL_DIM), dtype=torch.float32, device=dev)

    if S == 0:
        return freqs_cos, freqs_sin

    BLOCK_S = 8
    grid = (triton.cdiv(S, BLOCK_S),)
    _rope_kernel[grid](
        ids, powv, freqs_cos, freqs_sin, S,
        BLOCK_S=BLOCK_S, num_warps=4,
    )
    return freqs_cos, freqs_sin
