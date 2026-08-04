"""Fused Add + RMSNorm (hidden_size=2048, bf16) for MI355X / gfx950.

Semantics follow reference.py exactly:
    x   = hidden_states.f32 + residual.f32
    y   = (x * rsqrt(mean(x^2) + 1e-6)) * weight.f32
    out = y.bf16
Every intermediate is float32 and the result is rounded to bfloat16 exactly
once, at the store, matching the reference's single trailing `.to(dtype)`.

Three things make this fast:

 1. One program per row, the whole 2048-wide row held in registers. The add,
    the sum-of-squares reduction and the scale happen in a single pass, so each
    element of hidden_states and residual is read exactly once and each output
    element written once: 3 x 2 bytes per element, the memory-traffic lower
    bound for this operation.

 2. Streaming cache modifiers. The row data is single-use, so it is loaded
    `.cg` and stored `.cs` to keep it from evicting the weight vector, which
    every program reads. Worth ~9% on the large shapes.

 3. A pre-resolved launch path. Five of the seven workloads have 1..79 rows,
    where the kernel is ~2 us and Triton's per-call argument binding,
    specialization hashing and cache lookup cost more than the work does. We
    compile once via `warmup()` and then call the resulting CompiledKernel's
    launcher directly -- the same launcher the normal `kernel[grid](...)` path
    ends at, with the same arguments; only the re-derivation of an
    already-known answer is skipped. Anything that would invalidate the
    pre-resolved choice (non-contiguous inputs, pointers not 16-byte aligned,
    or installed launch hooks) falls back to the ordinary JIT path.

Measured on MI355X with rotating cold buffers, which is what the harness sees;
a single hot buffer flatters the large shapes by ~10% and misled an earlier
round of tuning here.
"""

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver

HIDDEN = 2048
# constexpr so the @jit kernel may reference it as a module global
EPS = tl.constexpr(1e-6)


@triton.jit
def _add_rmsnorm_row(H, R, W, O, HIDDEN: tl.constexpr):
    """One program == one row. Grid is exactly n_rows, so no bounds check.

    `weight` is deliberately left at default caching: it is the one input that
    every program reads, so it should stay resident.
    """
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, HIDDEN)
    off = row * HIDDEN + cols

    x = (tl.load(H + off, cache_modifier=".cg").to(tl.float32)
         + tl.load(R + off, cache_modifier=".cg").to(tl.float32))
    var = tl.sum(x * x, axis=0) * (1.0 / HIDDEN)
    inv = tl.rsqrt(var + EPS)
    y = (x * inv) * tl.load(W + cols).to(tl.float32)

    tl.store(O + off, y.to(tl.bfloat16), cache_modifier=".cs")


# (num_warps, waves_per_eu) by grid size, both measured on cold buffers:
#   small grids are launch-bound, so the config barely matters -- 4 warps;
#   large grids do best with one warp per row and waves_per_eu=2, trading
#   per-row parallelism for more rows in flight per EU.
_CFG_SMALL = (4, 0)
_CFG_LARGE = (1, 2)
_LARGE_THRESHOLD = 1024

_launchers = {}

# Device index, resolved on first use. Looking it up costs ~0.25 us per call,
# which is not nothing against a ~2 us kernel. Cached rather than captured at
# import time so it is resolved after the caller has selected its device.
_device = None


def _build(cfg):
    """Compile for (num_warps, waves_per_eu) and hoist the launcher pieces."""
    num_warps, waves_per_eu = cfg
    dummy = torch.empty(1, HIDDEN, device="cuda", dtype=torch.bfloat16)
    dummy_w = torch.empty(HIDDEN, device="cuda", dtype=torch.bfloat16)
    ck = _add_rmsnorm_row.warmup(
        dummy, dummy, dummy_w, dummy, HIDDEN,
        num_warps=num_warps, num_stages=1, waves_per_eu=waves_per_eu, grid=(1,),
    )
    ck._init_handles()
    entry = (ck.run, ck.function, ck.packed_metadata)
    _launchers[cfg] = entry
    return entry


def _hooks_active():
    """True if any Triton launch hook is installed (profiling/instrumentation).

    The hook slots hold `HookChain` objects that are always non-None, so an
    `is not None` test would be wrong -- an empty chain means nothing is
    listening. Treat anything that is not a recognisably-empty chain as active
    and route through the normal JIT path so the hooks actually fire.

    Read through `knobs` on every call rather than caching the chain objects:
    a profiler may replace the slot itself, not just append to it, and a stale
    reference would silently keep taking the fast path.
    """
    for hook in (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook):
        if hook is None:
            continue
        calls = getattr(hook, "calls", None)
        if calls is None or len(calls) > 0:
            return True
    return False


def run(hidden_states, residual, weight):
    # No @torch.no_grad() here: a raw Triton launch performs no autograd-tracked
    # op, so the output already comes back with requires_grad=False and no
    # grad_fn. The decorator only added ~1.5 us of guard setup per call, which
    # is significant against a ~2 us kernel.
    n_rows, hidden_size = hidden_states.shape
    assert hidden_size == HIDDEN

    if n_rows == 0:
        return torch.empty_like(hidden_states)

    # Rows are indexed linearly, so the kernel needs contiguous inputs.
    if not (hidden_states.is_contiguous() and residual.is_contiguous()
            and weight.is_contiguous()):
        hidden_states = hidden_states.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()

    out = torch.empty_like(hidden_states)

    cfg = _CFG_LARGE if n_rows >= _LARGE_THRESHOLD else _CFG_SMALL
    entry = _launchers.get(cfg)
    if entry is None:
        entry = _build(cfg)

    # The compiled kernel was specialized assuming 16-byte-aligned pointers.
    # True for any ordinary torch allocation, but not for an arbitrary offset
    # view, so check before using the pre-resolved launcher.
    if not ((hidden_states.data_ptr() | residual.data_ptr()
             | weight.data_ptr() | out.data_ptr()) & 15) and not _hooks_active():
        active = driver.active
        global _device
        dev = _device
        # Re-resolve if unset or if the caller has moved to another device;
        # `out` was just allocated on the current device, so its index is the
        # authority. The compare is a cheap attribute read.
        if dev is None or dev != out.device.index:
            dev = _device = active.get_current_device()
        entry[0](
            n_rows, 1, 1,
            active.get_current_stream(dev),
            entry[1], entry[2],
            None, None, None,
            hidden_states, residual, weight, out, HIDDEN,
        )
    else:
        _add_rmsnorm_row[(n_rows,)](
            hidden_states, residual, weight, out, HIDDEN,
            num_warps=cfg[0], num_stages=1, waves_per_eu=cfg[1],
        )

    return out
