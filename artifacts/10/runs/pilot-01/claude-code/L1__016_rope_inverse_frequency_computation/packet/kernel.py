import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# head_dim is a compile-time constant of this problem (128) -> 64 outputs.
_HEAD_DIM = 128
_HALF = _HEAD_DIM // 2


@triton.jit
def _inv_freq_kernel(out_ptr, theta, HALF: tl.constexpr):
    """
    inv_freq[i] = 1.0 / theta ** ((2*i) / 128)   for i in [0, 64)

    One workgroup, one wavefront, no intermediate global traffic: the whole
    thing is a single 256-byte store. The reference materialises three
    temporaries (indices, exponents, theta_powers) through HBM and launches
    four kernels; this launches one.

    NUMERICS -- the reason for enable_fp_fusion=False at the launch site.

    The reference is:
        indices   = arange(0, 128, 2, f32)          # exact
        exponents = indices / 128.0                 # exact: even int < 2^24
                                                    #        over a power of two
        powers    = torch.pow(theta, exponents)     # -> __ocml_pow_f32
        inv_freq  = 1.0 / powers                    # IEEE f32 divide

    torch's f32 pow on ROCm lowers to OCML's __ocml_pow_f32, and
    libdevice.pow binds the same symbol, so the two agree *bit for bit* --
    but only if the OCML bitcode is left alone. With fp contraction enabled
    (Triton's default) LLVM fuses the mul/add pairs inside OCML's
    log2/exp2 polynomials into FMAs. That changes the intermediate rounding
    and the result drifts by ~20 ulp. This problem's tolerance is ~1 ulp
    (rtol 1.19e-7 == f32 eps) with every one of the 64 elements required to
    match, so the fused build fails. With fusion off, all 16 workloads are
    bit-identical to the reference (max_abs == 0.0).

    Measured on gfx950: fused -> 808/1024 elements differ, max rel 1.6e-6;
    unfused -> 0/1024 differ.
    """
    i = tl.arange(0, HALF)
    exponents = (2.0 * i.to(tl.float32)) / 128.0
    powers = libdevice.pow(theta, exponents)
    tl.store(out_ptr + i, 1.0 / powers)


# ---------------------------------------------------------------------------
# Launch path.
#
# The kernel itself is ~1 us of work on 64 elements; everything else is
# dispatch cost. Triton's normal `kernel[grid](...)` path costs ~8 us per
# call in Python (bookkeeping, autotuner hooks, launch-metadata construction,
# hook lookups). We compile the kernel once at import time and then call the
# generated C launcher directly, which is ~3 us. Same kernel, same arguments,
# same stream -- only the Python-side overhead is removed.
#
# If any of this introspection fails (different Triton build), we fall back to
# the ordinary launch path, which is correct just slower.
# ---------------------------------------------------------------------------

_F32 = torch.float32
_DEV = torch.device("cuda", 0)

# torch.cuda.current_stream().cuda_stream costs ~2.5 us per call -- it builds a
# Python Stream object every time. The C accessor underneath is ~0.06 us and
# returns the same handle. Query it fresh on every call (do NOT cache the
# value: the harness may time on a non-default stream).
try:
    from torch._C import _cuda_getCurrentRawStream as _raw_stream
except ImportError:  # pragma: no cover
    def _raw_stream(_dev):
        return torch.cuda.current_stream().cuda_stream

_LAUNCH = None
_FUNCTION = None
_PACKED_META = None


def _build():
    global _LAUNCH, _FUNCTION, _PACKED_META
    scratch = torch.empty(_HALF, dtype=_F32, device=_DEV)
    compiled = _inv_freq_kernel.warmup(
        scratch,
        1.0e6,
        HALF=_HALF,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
        grid=(1,),
    )
    compiled._init_handles()
    _LAUNCH = compiled.run.launch
    _FUNCTION = compiled.function
    _PACKED_META = compiled.packed_metadata

    # Prove the fast path is wired up correctly before we commit to it:
    # run it once and require bit-equality with the reference formula.
    probe = torch.empty(_HALF, dtype=_F32, device=_DEV)
    _LAUNCH(False, 1, 1, 1, _raw_stream(0),
            _FUNCTION, None, _PACKED_META, None, None, None, probe, 1.0e6, _HALF)
    expect = torch.pow(
        1.0e6,
        torch.arange(0, _HEAD_DIM, 2, dtype=_F32, device=_DEV) / float(_HEAD_DIM),
    ).reciprocal()
    if not bool((probe == expect).all()):
        raise RuntimeError("fast launch path mismatch")


try:
    _build()
except Exception:  # pragma: no cover - fallback for other Triton builds
    _LAUNCH = None


def _make_run():
    """
    Bind everything run() touches into local cells at import time.

    At 64 elements the GPU work is a rounding error and the whole cost is
    Python. Each global/attribute lookup on the hot path is worth ~0.1-0.3 us
    against a ~4 us total, so the launch arguments that never change are
    frozen into a closure and the two that do (the output tensor and theta)
    are all that get built per call.

    There is no @torch.no_grad() here on purpose: the body runs no autograd
    tracked ops -- torch.empty on a leaf and a raw kernel launch -- so the
    decorator would buy nothing and its context manager costs ~1 us per call.
    """
    launch, function, meta = _LAUNCH, _FUNCTION, _PACKED_META
    empty, f32, dev, half = torch.empty, _F32, _DEV, _HALF
    stream = _raw_stream

    def run(rope_theta: float) -> torch.Tensor:
        out = empty(half, dtype=f32, device=dev)
        launch(False, 1, 1, 1, stream(0), function, None, meta,
               None, None, None, out, rope_theta, half)
        return out

    return run


def _make_run_fallback():
    @torch.no_grad()
    def run(rope_theta: float) -> torch.Tensor:
        out = torch.empty(_HALF, dtype=_F32, device=_DEV)
        _inv_freq_kernel[(1,)](
            out, float(rope_theta), HALF=_HALF,
            num_warps=1, num_stages=1, enable_fp_fusion=False,
        )
        return out

    return run


run = _make_run() if _LAUNCH is not None else _make_run_fallback()
