"""Fused Add + RMSNorm (hidden_size=2048, bf16) for MI355X / gfx950.

y = rmsnorm(f32(hidden_states) + f32(residual)) * f32(weight), cast to bf16.

The op is purely memory bound: 3 * batch * 2048 * 2 bytes must move (two reads,
one write), so a single fused Triton pass over the data is the whole algorithm
and the large-batch shapes land at HBM bandwidth. At small batch the kernel
finishes in well under a microsecond and total time is host-side launch cost,
so the hot path invokes the already-compiled binary directly, skipping Triton's
per-call argument binder and cache-key hashing.
"""

import torch
import triton
import triton.language as tl

H = 2048
EPS = tl.constexpr(1e-6)


@triton.jit(do_not_specialize=["n_rows"])
def _fused_add_rmsnorm(
    HS, RES, W, OUT, n_rows,
    ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    w = tl.load(W + cols).to(tl.float32)

    rows = pid * ROWS + tl.arange(0, ROWS)
    m2 = (rows < n_rows)[:, None]
    off = rows[:, None] * BLOCK + cols[None, :]

    # fp32 accumulation throughout, matching the reference's operation order:
    #   x = f32(h) + f32(res)
    #   inv = rsqrt(mean(x*x) + eps)
    #   y = (x * inv) * f32(w)   ->  bf16
    h = tl.load(HS + off, mask=m2, other=0.0).to(tl.float32)
    s = tl.load(RES + off, mask=m2, other=0.0).to(tl.float32)
    x = h + s
    m = tl.sum(x * x, axis=1) * (1.0 / BLOCK)
    y = (x * tl.rsqrt(m + EPS)[:, None]) * w[None, :]
    tl.store(OUT + off, y.to(OUT.dtype.element_ty), mask=m2)


# --- direct-launch fast path -------------------------------------------------
# The exact same compiled binary is invoked with the exact same arguments; only
# Triton's host-side dispatch layer is bypassed. All work still happens inside
# the launch, on the caller's stream, within the timed region.
_CACHE = {}
_NUM_WARPS = 4


def _build(rows, dev):
    x = torch.empty((rows, H), device=dev, dtype=torch.bfloat16)
    w = torch.empty((H,), device=dev, dtype=torch.bfloat16)
    o = torch.empty_like(x)
    compiled = _fused_add_rmsnorm[(1,)](
        x, x, w, o, rows, ROWS=rows, BLOCK=H,
        num_warps=_NUM_WARPS, num_stages=1,
    )
    try:
        compiled._init_handles()
        drv = triton.runtime.driver.active
        return (compiled.run, compiled.function, compiled.packed_metadata,
                drv.get_current_stream, drv.get_current_device)
    except Exception:
        return None


def _slow(hidden_states, residual, weight, out, n_rows, rows, grid0):
    """General path: lets Triton do its own specialization. Handles strided
    inputs, odd alignment, and anything the cached binary was not built for."""
    hs = hidden_states.contiguous().to(torch.bfloat16)
    rs = residual.contiguous().to(torch.bfloat16).expand_as(hs).contiguous()
    wt = weight.contiguous().to(torch.bfloat16)
    _fused_add_rmsnorm[(grid0,)](
        hs, rs, wt, out, n_rows, ROWS=rows, BLOCK=H,
        num_warps=_NUM_WARPS, num_stages=1,
    )
    return out


def _torch_fallback(hidden_states, residual, weight):
    x = hidden_states.to(torch.float32) + residual.to(torch.float32)
    inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return ((x * inv) * weight.to(torch.float32)).to(hidden_states.dtype)


def run(hidden_states, residual, weight):
    shape = hidden_states.shape
    if shape[1] != H:
        # hidden_size is a const axis (2048) in this definition; anything else
        # is outside what the kernel is compiled for.
        return _torch_fallback(hidden_states, residual, weight)
    n_rows = shape[0]
    out = torch.empty_like(hidden_states)
    if n_rows == 0:
        return out

    # One row per program up to the point where launch cost stops mattering,
    # then two rows so the grid does not outgrow what the dispatcher likes.
    if n_rows > 2048:
        rows = 2
        grid0 = (n_rows + 1) >> 1
    else:
        rows = 1
        grid0 = n_rows

    dev = hidden_states.device
    key = (rows, dev.index)
    try:
        ent = _CACHE[key]
    except KeyError:
        ent = _CACHE[key] = _build(rows, dev)

    # The cached binary is specialized on bf16 contiguous pointers; anything
    # else takes the general path rather than being reinterpreted.
    if ent is None or hidden_states.dtype is not torch.bfloat16 \
            or residual.dtype is not torch.bfloat16 \
            or weight.dtype is not torch.bfloat16 \
            or residual.shape != hidden_states.shape \
            or not (hidden_states.is_contiguous()
                    and residual.is_contiguous()
                    and weight.is_contiguous()):
        return _slow(hidden_states, residual, weight, out, n_rows, rows, grid0)

    # Resolve the live stream each call so we always enqueue where the caller
    # expects (side-stream capture, non-default streams, graph capture).
    stream = ent[3](ent[4]())
    ent[0](grid0, 1, 1, stream, ent[1], ent[2], None, None, None,
           hidden_states, residual, weight, out, n_rows, rows, H)
    return out
