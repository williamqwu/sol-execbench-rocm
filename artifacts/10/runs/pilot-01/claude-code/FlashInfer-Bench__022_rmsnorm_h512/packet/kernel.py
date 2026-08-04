"""RMSNorm (hidden_size=512, bf16) for MI355X / gfx950.

Two separate costs dominate this problem, and they need different fixes:

  * Large batch (11949, 14521 rows): pure streaming traffic, 4 B/element.
    One fused pass with non-temporal (".cs") stores lands ~4.0 us against a
    3.7 us HBM bound at 8 TB/s -- about 92% of Speed-of-Light.

  * Small batch: six of the eight workloads are <= 539 rows, where the kernel
    itself is ~2 us and the *CPU-side launch* is the real cost. Triton's Python
    dispatch is ~10.5 us/call; calling the compiled kernel's launcher directly
    is ~3.6 us. Measured end to end (async launch loop) that is 13.5 us -> 7.2 us
    per call.

The direct-launch path uses a private Triton ABI, so it is guarded three ways:
it is only used for contiguous inputs, every compiled variant is verified
against the ordinary JIT path the first time it is used, and any failure or
mismatch permanently falls back to the public API. Correctness never depends on
the fast path being available.

Numerics follow the reference exactly: fp32 accumulate, mean by multiplying by
1/512 (exact, a power of two), rsqrt(var + 1e-6), then (x * inv_rms) * weight
in fp32 with a single round to bf16 at the very end.
"""

import torch
import triton
import triton.language as tl

H = 512
EPS = 1e-6


@triton.jit
def _rmsnorm_kernel(
    X, W, Y,
    n_rows,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    EVEN: tl.constexpr,
):
    # Inputs are contiguous, so the row stride is exactly H and is folded in as
    # a constant -- this also keeps Triton's pointer-divisibility specialization
    # from depending on a runtime stride value.
    pid = tl.program_id(0)
    cols = tl.arange(0, H)
    w = tl.load(W + cols).to(tl.float32)

    rows = pid * ROWS + tl.arange(0, ROWS)
    off = rows[:, None] * H + cols[None, :]

    if EVEN:
        x = tl.load(X + off).to(tl.float32)
    else:
        mask = rows[:, None] < n_rows
        x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)

    # mean(x^2) in fp32; 1/H is a power of two so this multiply is exact.
    var = tl.sum(x * x, axis=1) * (1.0 / H)
    rstd = tl.rsqrt(var + 1e-6)

    y = ((x * rstd[:, None]) * w[None, :]).to(Y.dtype.element_ty)

    # The output is never re-read; keep it from evicting anything useful.
    if EVEN:
        tl.store(Y + off, y, cache_modifier=".cs")
    else:
        tl.store(Y + off, y, mask=rows[:, None] < n_rows, cache_modifier=".cs")


# (rows/program, num_warps), picked by an on-device sweep. Large batch is
# bandwidth bound and prefers 4 rows with one warp; small batch is launch bound
# and the tile choice barely matters.
_LARGE = (4, 1)
_SMALL = (1, 2)
_LARGE_THRESHOLD = 2048

_variants = {}
_direct_ok = True

_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _launch_jit(x, w, out, n, rows, warps, even, grid):
    _rmsnorm_kernel[(grid,)](
        x, w, out, n,
        ROWS=rows, H=H, EVEN=even,
        num_warps=warps, num_stages=1,
    )


def _build(rows, warps, even, device):
    """Compile a variant and prove the direct launcher matches the JIT path.

    Returns a callable (x, w, out, n, grid) -> None, or None if the direct ABI
    cannot be trusted for this variant.
    """
    if _raw_stream is None:
        return None

    dev = f"cuda:{device}"
    # Shape the probe so the masked variant actually exercises its mask.
    n_probe = rows * 3 - (1 if not even else 0)
    if n_probe < 1:
        n_probe = rows
    grid = (n_probe + rows - 1) // rows

    g = torch.Generator(device=dev).manual_seed(1234)
    xp = torch.randn((n_probe, H), device=dev, dtype=torch.bfloat16, generator=g)
    wp = torch.randn((H,), device=dev, dtype=torch.bfloat16, generator=g)

    expect = torch.empty_like(xp)
    _launch_jit(xp, wp, expect, n_probe, rows, warps, even, grid)
    torch.cuda.synchronize()

    try:
        ck = _rmsnorm_kernel.warmup(
            xp, wp, expect, n_probe,
            ROWS=rows, H=H, EVEN=even,
            num_warps=warps, num_stages=1, grid=(grid,),
        )
        ck._init_handles()

        fn, pmd, launch = ck.function, ck.packed_metadata, ck.run

        def direct(x, w, out, n, grid, _l=launch, _f=fn, _m=pmd):
            _l(grid, 1, 1, _raw_stream(x.device.index), _f, _m, None, None, None,
               x, w, out, n, rows, H, even)

        got = torch.empty_like(xp)
        direct(xp, wp, got, n_probe, grid)
        torch.cuda.synchronize()

        if not torch.equal(got, expect):
            return None
        return direct
    except Exception:
        return None


def _capturing():
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _get(rows, warps, even, device):
    key = (rows, warps, even, device)
    if key in _variants:
        return _variants[key]
    # Building a variant compiles and synchronizes, neither of which is legal
    # mid graph-capture. Skip it this once (without caching the miss) and use
    # the public API; the variant gets built on a later, uncaptured call.
    if _capturing():
        return None
    _variants[key] = _build(rows, warps, even, device)
    return _variants[key]


def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 512

    x = hidden_states
    if not x.is_contiguous():
        x = x.contiguous()
    w = weight
    if w.stride(0) != 1:
        w = w.contiguous()

    out = torch.empty_like(x)
    if batch_size == 0:
        return out

    rows, warps = _LARGE if batch_size > _LARGE_THRESHOLD else _SMALL
    grid = (batch_size + rows - 1) // rows
    even = (batch_size % rows) == 0

    global _direct_ok
    if _direct_ok:
        try:
            direct = _get(rows, warps, even, x.device.index)
            if direct is not None:
                direct(x, w, out, batch_size, grid)
                return out
        except Exception:
            _direct_ok = False

    _launch_jit(x, w, out, batch_size, rows, warps, even, grid)
    return out
