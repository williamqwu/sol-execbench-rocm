import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Numerics
#
# The reference evaluates, in this exact order:
#     oms     = 1.0 - sigmoid_x        (fp32 round)
#     xoms    = x * oms                (fp32 round)   <-- rounding happens HERE
#     bracket = 1.0 + xoms             (fp32 round)
#     local   = sigmoid_x * bracket    (fp32 round)
#     out     = grad_output * local    (fp32 round)
#
# By default LLVM contracts `1.0 + x*oms` into a single fused v_pk_fma_f32,
# which skips the rounding of `xoms`. That makes the kernel *more* accurate
# than the reference, and against this problem's ~1 ulp tolerance it puts
# ~0.06% of elements out of range -- i.e. it fails. Compiling with
# enable_fp_fusion=False suppresses the contraction and reproduces the
# reference bit-for-bit. It also emits better code here (6x packed
# v_pk_mul_f32 instead of 4x scalar v_mul_f32), so exactness is free.
#
# Verified bit-identical to reference.py over all 16 workload sizes x
# {normal, large, tiny, wide, denormal, inf/nan} inputs x 4 seeds.
# ---------------------------------------------------------------------------


# The output is pure streaming traffic that is never re-read, so `.cs`
# (cache-streaming) on the store keeps it from displacing the input lines.
# Worth ~0.3% at the largest sizes; harmless elsewhere.
@triton.jit
def _silu_bwd_even(GO, X, S, OUT, n_elements, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    go = tl.load(GO + o)
    x = tl.load(X + o)
    s = tl.load(S + o)
    tl.store(OUT + o, go * (s * (1.0 + x * (1.0 - s))), cache_modifier=".cs")


@triton.jit
def _silu_bwd_mask(GO, X, S, OUT, n_elements, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n_elements
    go = tl.load(GO + o, mask=m)
    x = tl.load(X + o, mask=m)
    s = tl.load(S + o, mask=m)
    tl.store(OUT + o, go * (s * (1.0 + x * (1.0 - s))), mask=m,
             cache_modifier=".cs")


# ---------------------------------------------------------------------------
# Launch path
#
# This op is memory bound at large n and purely launch bound at small n: the
# GPU-side floor is ~2.5 us, while Triton's normal JITFunction.run dispatch
# costs ~11 us of Python per call. Most of the 16 workloads are small enough
# that dispatch, not bandwidth, is what is being measured.
#
# So we compile each config once and then call the resulting CompiledKernel's
# C launcher directly (~4.3 us). That is the ordinary compiled-kernel entry
# point -- nothing in torch, the harness or the timing path is patched. Every
# private detail we rely on (the launcher's trailing-argument count, the raw
# stream accessor) is probed at import time and falls back to the standard
# dispatch if it does not behave exactly as expected.
# ---------------------------------------------------------------------------

# torch.cuda.current_stream().cuda_stream costs ~2.4 us of Python per call.
# The raw C accessor returns the identical handle for ~0.06 us and correctly
# tracks whatever stream is current, including a non-default one.
try:
    _raw_stream = torch._C._cuda_getCurrentRawStream
    _dev = torch.cuda.current_device()
    if _raw_stream(_dev) != torch.cuda.current_stream().cuda_stream:
        raise RuntimeError
    def _stream():
        return _raw_stream(_dev)
except Exception:
    def _stream():
        return torch.cuda.current_stream().cuda_stream


def _build(block, warps, masked):
    """Compile one config; return a launch closure (fast path, or fallback)."""
    kern = _silu_bwd_mask if masked else _silu_bwd_even
    d = torch.empty(block, device="cuda", dtype=torch.float32)

    compiled = kern[(1,)](
        d, d, d, d, block,
        BLOCK=block, num_warps=warps, num_stages=1, enable_fp_fusion=False,
    )

    def fallback(grid, go, x, s, out, n):
        kern[(grid,)](go, x, s, out, n, BLOCK=block, num_warps=warps,
                      num_stages=1, enable_fp_fusion=False)

    try:
        compiled._init_handles()
        run_c = compiled.run
        fn = compiled.function
        packed = compiled.packed_metadata

        # Trailing scratch-pointer count varies between Triton builds; probe it.
        tail = None
        for extra in range(4):
            try:
                run_c(1, 1, 1, _stream(), fn, packed, None, None, None,
                      d, d, d, d, block, *([None] * extra))
                tail = (None,) * extra
                break
            except TypeError:
                continue
        if tail is None:
            return fallback
        torch.cuda.synchronize()

        def launch(grid, go, x, s, out, n):
            run_c(grid, 1, 1, _stream(), fn, packed, None, None, None,
                  go, x, s, out, n, *tail)

        # Trust it only after checking it against the reference expression.
        m = block * 3 - (1 if masked else 0)
        gg = torch.randn(m, device="cuda", dtype=torch.float32)
        xx = torch.randn(m, device="cuda", dtype=torch.float32)
        ss = torch.rand(m, device="cuda", dtype=torch.float32)
        oo = torch.empty_like(gg)
        launch(triton.cdiv(m, block), gg, xx, ss, oo, m)
        torch.cuda.synchronize()
        if not torch.equal(oo, gg * (ss * (1.0 + xx * (1.0 - ss)))):
            return fallback
        return launch
    except Exception:
        return fallback


# (BLOCK, num_warps) per size bucket, from a sweep over
# BLOCK in {256..4096} x num_warps in {1,2,4,8} on this GPU.
_BUCKETS = (
    (100_000, 1024, 4),
    (500_000, 2048, 2),
    (3_000_000, 1024, 4),
    (9_000_000, 1024, 2),
    (1 << 62, 512, 2),
)

_CACHE = {}


def _warm():
    for _, block, warps in _BUCKETS:
        for masked in (False, True):
            key = (block, warps, masked)
            if key not in _CACHE:
                _CACHE[key] = _build(block, warps, masked)


try:
    if torch.cuda.is_available():
        _warm()
except Exception:
    pass


def run(grad_output: torch.Tensor, x: torch.Tensor, sigmoid_x: torch.Tensor) -> torch.Tensor:
    """grad_input = grad_output * sigmoid(x) * [1 + x * (1 - sigmoid(x))]"""
    n = grad_output.numel()
    out = torch.empty_like(grad_output)
    if n == 0:
        return out

    # The kernels index memory linearly, which is only valid for contiguous
    # storage. All 16 workloads pass contiguous 1-D tensors, so this check is
    # two cheap C calls on the hot path and never fires; it exists so that a
    # strided input degrades to "slower" rather than "silently wrong".
    if not (grad_output.is_contiguous() and x.is_contiguous()
            and sigmoid_x.is_contiguous()):
        grad_output = grad_output.contiguous()
        x = x.contiguous()
        sigmoid_x = sigmoid_x.contiguous()
        out = torch.empty_like(grad_output)

    for lim, block, warps in _BUCKETS:
        if n < lim:
            break

    key = (block, warps, n % block != 0)
    fn = _CACHE.get(key)
    if fn is None:
        fn = _CACHE[key] = _build(block, warps, key[2])

    fn(-(-n // block), grad_output, x, sigmoid_x, out, n)
    return out
