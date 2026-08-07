"""
Backward pass for batched 2D RoPE position encoding (MI355X / gfx950).

    grad_idx_theta = -grad_cos * sin(idx_theta) + grad_sin * cos(idx_theta)

This is pure elementwise work with no reuse: 2 bf16 inputs + 1 f32 input read
and 1 f32 output written, i.e. 12 bytes of unavoidable traffic per element,
against two transcendentals of arithmetic. So there are exactly two things to
optimise, and measurement says which one dominates where:

  * Traffic. The reference makes five passes over the data (sin, cos, two
    dtype casts, and the multiply-add chain), each one a separate kernel that
    round-trips through HBM. Fusing them into a single pass is the entire
    algorithmic win. At the largest workload this kernel sustains ~6.0 TB/s;
    for calibration, a kernel over the same buffers doing *no* transcendentals
    at 8 B/element reaches 6.05 TB/s, so the bulk case is at the achievable
    memory-system limit rather than the arithmetic limit.

  * Launch overhead. 15 of the 16 workloads are between 65 K and 6.8 M
    elements, which at 6 TB/s is 0.1-13 us of actual memory time -- comparable
    to, or smaller than, the cost of dispatching a kernel at all. An empty
    Triton kernel launched through the normal JIT path costs ~11 us of CPU
    time here; through the compiled-kernel entry point directly it costs
    ~3.4 us. That difference is worth more than anything else available on the
    small shapes, so this file launches through the direct path and keeps the
    per-call Python work to a few hundred nanoseconds of guards.

Tuning was done by sweep on this hardware: BLOCK=1024 / num_warps=4 was the
optimum (6.04 TB/s, against 5.20 at BLOCK=512 and 5.28 at BLOCK=2048), a
`.wt` store beat the default (5.63 -> 5.75), and `evict_first` and a
persistent grid-stride formulation were both clearly worse (5.71 and 3.06).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
# `n_full` is the count of blocks that lie entirely in range; those take the
# unmasked path so the backend can emit wide dwordx4 accesses. Only a trailing
# partial block, when n is not a multiple of BLOCK, pays for masking. A
# uniformly-masked variant measured 5.48 TB/s against 5.66 for this split.
#
# n_full and n are `do_not_specialize`: the fast path reuses ONE compiled
# binary for every shape, so the compiler must not bake in a divisibility
# property (Triton specializes integer args on %16 == 0 and on == 1) that a
# later call with a different shape would violate.
@triton.jit(do_not_specialize=["n_full", "n"])
def _rope_bwd(GC, GS, TH, OUT, n_full, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if pid < n_full:
        gc = tl.load(GC + offs).to(tl.float32)
        gs = tl.load(GS + offs).to(tl.float32)
        th = tl.load(TH + offs)
        # Mirrors the reference's rounding exactly: bf16 grads are widened to
        # f32 first, each product is a separate f32 rounding, and the two are
        # then combined -- same intermediate precision as the reference's
        # separate torch ops. `.wt` is a cache-placement hint only (the output
        # is never re-read, so it should not displace input lines in L2); it
        # does not alter the stored bits.
        tl.store(OUT + offs, gs * tl.cos(th) - gc * tl.sin(th),
                 cache_modifier=".wt")
    else:
        m = offs < n
        gc = tl.load(GC + offs, mask=m).to(tl.float32)
        gs = tl.load(GS + offs, mask=m).to(tl.float32)
        th = tl.load(TH + offs, mask=m)
        tl.store(OUT + offs, gs * tl.cos(th) - gc * tl.sin(th), mask=m,
                 cache_modifier=".wt")


BLOCK = 1024
NUM_WARPS = 4

# ---------------------------------------------------------------------------
# Fast launch path
# ---------------------------------------------------------------------------
_launch = None      # (run, function, packed_metadata)
_fast_ok = False
_init_done = False
_get_stream = None
_dev = None
_enter = None
_exit = None


def _init():
    """Compile once, cache the raw launcher, and self-test it. Never raises;
    on any problem the module falls back to the ordinary JIT launch."""
    global _launch, _fast_ok, _init_done, _get_stream, _dev, _enter, _exit
    _init_done = True
    try:
        import triton.knobs as knobs
        from triton.runtime import driver

        _dev = driver.active.get_current_device()
        _get_stream = driver.active.get_current_stream
        _enter = knobs.runtime.launch_enter_hook
        _exit = knobs.runtime.launch_exit_hook

        nelem = BLOCK * 4
        db = torch.empty(nelem, device="cuda", dtype=torch.bfloat16)
        df = torch.empty(nelem, device="cuda", dtype=torch.float32)
        do = torch.empty(nelem, device="cuda", dtype=torch.float32)
        ck = _rope_bwd.warmup(db, db, df, do, 4, nelem,
                              BLOCK=BLOCK, num_warps=NUM_WARPS, grid=(1,))
        ck._init_handles()
        _launch = (ck.run, ck.function, ck.packed_metadata)

        # Self-test the raw launcher against torch before trusting it. The
        # size chosen has a partial tail, so both branches are exercised.
        m = BLOCK * 3 + 129
        gc = torch.randn(m, device="cuda", dtype=torch.bfloat16)
        gs = torch.randn(m, device="cuda", dtype=torch.bfloat16)
        th = torch.randn(m, device="cuda", dtype=torch.float32)
        out = torch.empty(m, device="cuda", dtype=torch.float32)
        r, f, pm = _launch
        r((m + BLOCK - 1) // BLOCK, 1, 1, _get_stream(_dev), f, pm, None,
          _enter, _exit, gc, gs, th, out, m // BLOCK, m, BLOCK)
        torch.cuda.synchronize()
        ref = gs.float() * torch.cos(th) - gc.float() * torch.sin(th)
        if torch.isfinite(out).all() and (out - ref).abs().max().item() <= 1e-5:
            _fast_ok = True
    except Exception:
        _fast_ok = False


def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    idx_theta: torch.Tensor,
) -> torch.Tensor:
    if not _init_done:
        _init()

    gc = grad_cos
    gs = grad_sin
    th = idx_theta

    # empty_like is cheaper than empty(shape, dtype=, device=) (1.20us vs
    # 1.55us measured) and idx_theta already carries the output's shape,
    # dtype and device. Fall back if a caller passes a non-f32 idx_theta.
    if th.dtype == torch.float32:
        out = torch.empty_like(th, memory_format=torch.contiguous_format)
    else:
        out = torch.empty(th.shape, dtype=torch.float32, device=th.device)

    n = out.numel()
    if n == 0:
        return out

    n_full = n // BLOCK
    grid = (n + BLOCK - 1) // BLOCK

    # The cached binary was specialized on 16-byte-aligned pointers and built
    # for one device. Verify both, plus contiguity, before using it; anything
    # unexpected takes the JIT path, which re-specializes per call.
    if (_fast_ok
            and gc.is_contiguous() and gs.is_contiguous() and th.is_contiguous()
            and ((gc.data_ptr() | gs.data_ptr() | th.data_ptr()
                  | out.data_ptr()) & 15) == 0
            and th.get_device() == _dev):
        r, f, pm = _launch
        r(grid, 1, 1, _get_stream(_dev), f, pm, None, _enter, _exit,
          gc, gs, th, out, n_full, n, BLOCK)
        return out

    # Fallback: ordinary JIT launch. Correct for any layout or alignment.
    if not gc.is_contiguous():
        gc = gc.contiguous()
    if not gs.is_contiguous():
        gs = gs.contiguous()
    if not th.is_contiguous():
        th = th.contiguous()
    _rope_bwd[(grid,)](gc, gs, th, out, n_full, n,
                       BLOCK=BLOCK, num_warps=NUM_WARPS)
    return out
