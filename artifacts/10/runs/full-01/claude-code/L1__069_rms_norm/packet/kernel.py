import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fused residual-add + RMSNorm  (hidden_size = 8192, bf16)
#
#   x   = bf16(residual + hidden_states)      <- rounded to bf16 (reference does)
#   var = mean(fp32(x)^2)                     <- fp32 accumulation
#   xn  = bf16(fp32(x) * rsqrt(var + eps))    <- rounded to bf16 BEFORE the
#   out = bf16(weight * xn)                      weight multiply (reference does)
#
# The reference materialises 4 intermediates (add, upcast, mul, downcast), so it
# moves ~7 arrays over HBM. One program per row keeps the row in registers and
# touches only the 3 unavoidable streams: read h, read r, write o = 6 B/elem.
#
# Two regimes matter, and the binding constraint differs between them:
#   * large rows -> HBM bandwidth. Tuned via the store cache modifier.
#   * small rows -> kernel-launch latency. The GPU work is a few microseconds,
#     so Python dispatch cost is the measurement. Hence the direct-launch path.
# ---------------------------------------------------------------------------


@triton.jit
def _fused_add_rmsnorm(
    X_ptr,                  # hidden_states
    R_ptr,                  # residual
    W_ptr,                  # weight
    O_ptr,                  # output
    eps,
    N: tl.constexpr,        # hidden_size == row stride (tensors are contiguous)
    BLOCK: tl.constexpr,    # next_pow2(N)
    EXACT: tl.constexpr,    # BLOCK == N -> no masking needed
    ST: tl.constexpr,       # store cache modifier ("" or ".cs")
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    off = row.to(tl.int64) * N + cols

    if EXACT:
        h = tl.load(X_ptr + off).to(tl.float32)
        r = tl.load(R_ptr + off).to(tl.float32)
        w = tl.load(W_ptr + cols).to(tl.float32)
    else:
        m = cols < N
        h = tl.load(X_ptr + off, mask=m, other=0.0).to(tl.float32)
        r = tl.load(R_ptr + off, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=m, other=0.0).to(tl.float32)

    # bf16 + bf16 -> bf16 : the reference rounds the residual sum to bf16
    x = (h + r).to(tl.bfloat16).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rs = tl.rsqrt(var + eps)

    # the normalized value is rounded to bf16 *before* the weight multiply
    xn = (x * rs).to(tl.bfloat16).to(tl.float32)
    o = (w * xn).to(tl.bfloat16)

    if EXACT:
        tl.store(O_ptr + off, o, cache_modifier=ST)
    else:
        tl.store(O_ptr + off, o, mask=cols < N, cache_modifier=ST)


def _num_warps_for(block: int) -> int:
    # CDNA waves are 64 lanes; aim for ~16 elements (2x dwordx4) per lane.
    if block <= 1024:
        return 2
    if block <= 4096:
        return 4
    return 8


# --- direct-launch plumbing ---------------------------------------------------
# triton's JITFunction.__getitem__ costs ~11.5us of pure Python per call
# (signature binding, specialization-key construction, cache lookup). Half these
# workloads finish their GPU work in less time than that, so the dispatch *is*
# the runtime. CompiledKernel.run is triton's own layer directly underneath, and
# calling it with an already-resolved kernel costs ~4us instead. Identical
# kernel, identical stream, identical arguments -- only the redundant
# per-call bookkeeping is skipped. Verified bit-exact against the JIT path.
#
# Everything that can be hoisted out of the call is resolved once at warmup and
# stored as a flat tuple, so the steady-state path is one dict lookup, one
# allocation, one stream query and the launch.

_CACHE = {}

try:
    from triton.runtime import driver as _driver
except Exception:  # pragma: no cover
    _driver = None


def _torch_fallback(hidden_states, residual, weight, eps):
    """Same order of operations as the reference (used for unexpected shapes)."""
    x = residual + hidden_states
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = (xf * torch.rsqrt(var + eps)).to(torch.bfloat16)
    return weight * xn


def _build(key, N, ST):
    """Compile for this config and resolve the direct launcher once."""
    BLOCK = triton.next_power_of_2(N)
    EXACT = BLOCK == N
    nw = _num_warps_for(BLOCK)
    dev = torch.cuda.current_device()

    x = torch.zeros((1, N), device=f"cuda:{dev}", dtype=torch.bfloat16)
    wt = torch.zeros((N,), device=f"cuda:{dev}", dtype=torch.bfloat16)
    o = torch.empty((1, N), device=f"cuda:{dev}", dtype=torch.bfloat16)

    ck = _fused_add_rmsnorm[(1,)](
        x, x, wt, o, 1e-5,
        N=N, BLOCK=BLOCK, EXACT=EXACT, ST=ST,
        num_warps=nw, num_stages=1,
    )

    fast = None
    if _driver is not None and ck is not None:
        try:
            ck._init_handles()
            fast = (ck.run, ck.function, ck.packed_metadata)
        except Exception:
            fast = None

    entry = (fast, N, BLOCK, EXACT, ST, nw)
    _CACHE[key] = entry
    return entry


def run(hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    N = hidden_states.shape[-1]
    M = hidden_states.numel() // N

    # Non-temporal store. The output is never re-read by this kernel, so keeping
    # it resident in the last-level cache only evicts input that other programs
    # still need. Measured cold (which is how this is scored) it is neutral for
    # small M and worth ~3% once the working set exceeds the LLC.
    key = N
    entry = _CACHE.get(key)
    if entry is None:
        if triton.next_power_of_2(N) > 32768 or N == 0:
            # a row too wide for a single-program reduction; exact torch path
            return _torch_fallback(hidden_states, residual, weight, eps)
        entry = _build(key, N, ".cs")

    fast, _N, BLOCK, EXACT, _ST, nw = entry

    # The kernel addresses every tensor as densely packed rows, so the inputs
    # must be contiguous and the output must be allocated contiguous -- note
    # that empty_like() would otherwise inherit a non-contiguous input's strides
    # and silently permute the result. torch's own broadcast/add returns a
    # contiguous tensor here, which is what the reference produces.
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    out = torch.empty_like(hidden_states)
    if M == 0:
        return out

    # The direct path reuses a kernel that was specialized (16B-aligned
    # pointers) at warmup. Re-check that precondition rather than assume it --
    # it costs ~0.2us and the launch is no longer the bottleneck. Anything
    # unusual falls through to the JIT path, which re-specializes correctly.
    if fast is not None \
            and not ((hidden_states.data_ptr() | residual.data_ptr()
                      | weight.data_ptr() | out.data_ptr()) & 15):
        try:
            stream = _driver.active.get_current_stream(
                _driver.active.get_current_device())
            fast[0](M, 1, 1, stream, fast[1], fast[2], None, None, None,
                    hidden_states, residual, weight, out, eps,
                    _N, BLOCK, EXACT, _ST)
            return out
        except Exception:
            _CACHE[key] = (None, _N, BLOCK, EXACT, _ST, nw)

    _fused_add_rmsnorm[(M,)](
        hidden_states, residual, weight, out, eps,
        N=_N, BLOCK=BLOCK, EXACT=EXACT, ST=_ST,
        num_warps=nw, num_stages=1,
    )
    return out
