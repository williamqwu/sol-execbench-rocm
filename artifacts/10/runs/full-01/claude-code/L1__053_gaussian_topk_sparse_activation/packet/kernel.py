import numpy as np
import torch
import triton
import triton.language as tl
from functools import lru_cache


# ---------------------------------------------------------------------------
# Host-side inverse normal CDF (Abramowitz & Stegun 26.2.23).
#
# The reference evaluates this on a 0-d float32 CUDA tensor, so every
# intermediate is rounded to float32.  We reproduce that exact order of
# operations with numpy float32 scalars.  The central region -- the only one
# any workload hits, p in [0.02425, 0.97575] -- uses nothing but +,-,*,/ which
# are IEEE-exact in float32, so the z-score comes out bit-identical to the
# reference.  The tails use log/sqrt and land within 1 ulp.
#
# Doing this on the host removes ~10 tiny device kernels plus the two `.any()`
# host syncs the reference pays, and yields a plain Python float that folds
# into the main kernel as a scalar argument.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _ndtri_f32(p_in: float) -> float:
    f = np.float32

    a1 = f(-3.969683028665376e+01); a2 = f(2.209460984245205e+02)
    a3 = f(-2.759285104469687e+02); a4 = f(1.383577518672690e+02)
    a5 = f(-3.066479806614716e+01); a6 = f(2.506628277459239e+00)

    b1 = f(-5.447609879822406e+01); b2 = f(1.615858368580409e+02)
    b3 = f(-1.556989798598866e+02); b4 = f(6.680131188771972e+01)
    b5 = f(-1.328068155288572e+01)

    c1 = f(-7.784894002430293e-03); c2 = f(-3.223964580411365e-01)
    c3 = f(-2.400758277161838e+00); c4 = f(-2.549732539343734e+00)
    c5 = f(4.374664141464968e+00);  c6 = f(2.938163982698783e+00)

    d1 = f(7.784695709041462e-03);  d2 = f(3.224671290700398e-01)
    d3 = f(2.445134137142996e+00);  d4 = f(3.754408661907416e+00)

    one = f(1.0)
    p = f(p_in)
    p_low = f(0.02425)
    p_high = f(one - p_low)

    if p < p_low:
        q = np.sqrt(f(-2.0) * np.log(p))
        num = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = ((((d1 * q + d2) * q + d3) * q + d4) * q + one)
        return float(f(num / den))
    if p <= p_high:
        q = p - f(0.5)
        r = q * q
        num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + one)
        return float(f(num / den))
    q = np.sqrt(f(-2.0) * np.log(one - p))
    num = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
    den = ((((d1 * q + d2) * q + d3) * q + d4) * q + one)
    return float(f(-(num / den)))


# ---------------------------------------------------------------------------
# Fused kernel.  One program owns one row.  The row is read from HBM exactly
# once into registers; the mean, the two-pass variance, the thresholded ReLU
# and the bfloat16 round-trip all run on that resident copy.  Traffic is
# therefore the unavoidable minimum -- one read and one write of the tensor --
# which measures faster than torch's own device-to-device copy at the large
# shapes.
#
# D is a constexpr: the workloads use only a handful of feature widths, and
# making it compile-time lets the backend prove 16-byte alignment and emit
# wide dwordx4 accesses.  (Passing D as an opaque runtime int costs ~40% of
# bandwidth at D=12288 because that proof is lost.)
#
# Variance is two-pass -- sum of (x-mean)^2 -- matching what
# torch.std(unbiased=False) does.  The one-pass E[x^2]-mu^2 form is cheaper
# but loses precision to cancellation, and the reference's rounding is part of
# the spec here.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["ROWS"])
def _gtsa(X, Y, MUL, ROWS, D: tl.constexpr, BLOCK: tl.constexpr,
          SCM: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    EVEN: tl.constexpr = (BLOCK == D)

    if pid < ROWS:
        base = pid.to(tl.int64) * D
        if EVEN:
            x = tl.load(X + base + offs)
        else:
            x = tl.load(X + base + offs, mask=offs < D, other=0.0)

        xf = x.to(tl.float32)
        mean = tl.sum(xf, axis=0) * (1.0 / D)

        d = xf - mean
        if not EVEN:
            d = tl.where(offs < D, d, 0.0)
        var = tl.sum(d * d, axis=0) * (1.0 / D)

        out = tl.maximum(xf - (mean + tl.sqrt(var) * MUL), 0.0).to(tl.bfloat16)

        # The output is never re-read by this kernel, so a streaming store that
        # skips L2 write-allocate is strictly better: it leaves the cache to
        # the input stream instead of thrashing it with write-allocate lines.
        # Worth up to 28% -- 7.7 TB/s vs 5.5 -- and never measured as a loss.
        if EVEN:
            tl.store(Y + base + offs, out, cache_modifier=SCM)
        else:
            tl.store(Y + base + offs, out, mask=offs < D, cache_modifier=SCM)


# Store modifier and warp count, measured per (D, rows) on MI355X with launch
# cost excluded so they reflect GPU time rather than Triton dispatch.  Both
# ".cs" and ".wt" are non-temporal; which of the two wins tracks whether the
# working set still fits usefully in L2.
def _plan(D, rows):
    if D <= 4096:
        if rows < 1024:
            return "", 1
        return (".wt", 4) if rows < 4096 else (".cs", 4)
    if D <= 8192:
        if rows < 1024:
            return ".wt", 8
        if rows < 4096:
            return ".cs", 2
        return ".cs", 8
    if D <= 12288:
        return (".wt", 8) if rows < 1024 else (".cs", 8)
    return ".cs", 8


# ---------------------------------------------------------------------------
# Raw-launch fast path.
#
# Triton's `kernel[grid](...)` dispatch costs ~9.5us of pure Python per call on
# this stack; the precompiled launcher underneath it costs ~3.8us.  Seven of
# the twelve workloads are small enough that this gap, not memory traffic,
# sets their latency.  So we compile once per distinct configuration, keep the
# CompiledKernel, and thereafter invoke its launcher directly.
#
# This is Triton's own public launch path -- the same call that
# CompiledKernel.__getitem__ makes -- with the per-call Python re-derivation
# hoisted into a dict.  Nothing is patched or overridden.  If any part of it
# fails to line up (different Triton build, changed launcher arity), the entry
# caches as None and every later call falls back to the ordinary JIT path,
# which is always correct.
# ---------------------------------------------------------------------------
_CACHE = {}
_DRIVER = None
_DEV = None


def _driver():
    global _DRIVER, _DEV
    if _DRIVER is None:
        from triton.runtime import driver
        _DRIVER = driver.active
        _DEV = _DRIVER.get_current_device()
    return _DRIVER, _DEV


def _launch(fast, rows, D, block, scm, x, y, mul):
    """Invoke a cached kernel. `fast[5]` is the cooperative-grid flag when we
    bound the inner C launcher, or None when we kept the Python wrapper (whose
    signature omits both that flag and the profile-scratch slot)."""
    fn, func, pmd, get_stream, dev, cg = fast
    stream = get_stream(dev)
    if cg is None:
        fn(rows, 1, 1, stream, func, pmd,
           None, None, None, x, y, mul, rows, D, block, scm)
    else:
        fn(cg, rows, 1, 1, stream, func, None, pmd,
           None, None, None, x, y, mul, rows, D, block, scm)


def _build(key, D, block, scm, nw, rows, x, y, mul):
    """Compile through the JIT once, then extract the CompiledKernel and
    verify its launcher accepts our argument arity before relying on it."""
    try:
        drv, dev = _driver()
        cache = _gtsa.device_caches[dev][0]
        before = set(cache.keys())
        _gtsa[(rows,)](x, y, mul, rows, D, block, scm,
                       num_warps=nw, num_stages=1)
        new = set(cache.keys()) - before
        if len(new) != 1:
            _CACHE[key] = None
            return None
        ck = cache[new.pop()]
        ck._init_handles()

        # ck.run is a HIPLauncher whose __call__ re-derives a profile-scratch
        # allocation on every launch.  When there is no profile scratch and no
        # cooperative grid -- the case for this kernel -- that wrapper is pure
        # overhead (~0.4us of ~4.0us), so bind its inner C entry point directly
        # and pass the profile_scratch slot as None ourselves.  Guarded: if
        # either precondition fails we keep the wrapper.
        launcher = ck.run
        inner = getattr(launcher, "launch", None)
        if (inner is not None
                and getattr(launcher, "profile_scratch_size", 1) == 0
                and not getattr(launcher, "launch_cooperative_grid", True)):
            cg = launcher.launch_cooperative_grid
            fast = (inner, ck.function, ck.packed_metadata,
                    drv.get_current_stream, dev, cg)
        else:
            fast = (launcher, ck.function, ck.packed_metadata,
                    drv.get_current_stream, dev, None)

        # The stream is fetched per call, never cached: the harness may time on
        # a non-default stream, and launching on a stale one would be an
        # ordering bug that no tolerance check would catch. The lookup is 0.06us.
        _launch(fast, rows, D, block, scm, x, y, mul)
    except Exception:
        _CACHE[key] = None
        return None
    _CACHE[key] = fast
    return fast


# next_power_of_2 costs ~1us of the ~6.6us Python path; there are only a
# handful of distinct feature widths, so memoise it.
_POW2 = {}
_PLAN = {}
_empty = torch.empty
_empty_like = torch.empty_like


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    # The reference returns the input tensor itself when no sparsity is asked.
    if target_sparsity == 0.0:
        return inputs

    if not inputs.is_contiguous():
        inputs = inputs.contiguous()

    shape = inputs.shape
    D = shape[-1]
    numel = inputs.numel()
    if D == 0 or numel == 0:
        return _empty(shape, dtype=torch.bfloat16, device=inputs.device)

    rows = numel // D
    out = _empty_like(inputs)
    # float() both normalises the lru_cache key (a numpy scalar or 0-d tensor
    # would otherwise miss, or be unhashable) and matches the reference, which
    # builds a float32 tensor from this value.
    mul = _ndtri_f32(float(target_sparsity))

    pkey = (D, rows)
    plan = _PLAN.get(pkey)
    if plan is None:
        block = _POW2.get(D)
        if block is None:
            block = _POW2[D] = triton.next_power_of_2(D)
        scm, nw = _plan(D, rows)
        plan = _PLAN[pkey] = (block, scm, nw, (D, block, scm, nw))
    block, scm, nw, key = plan

    fast = _CACHE.get(key, 0)
    if fast == 0:  # never attempted for this configuration
        fast = _build(key, D, block, scm, nw, rows, inputs, out, mul)

    if fast is not None:
        _launch(fast, rows, D, block, scm, inputs, out, mul)
    else:
        _gtsa[(rows,)](inputs, out, mul, rows, D, block, scm,
                       num_warps=nw, num_stages=1)
    return out
