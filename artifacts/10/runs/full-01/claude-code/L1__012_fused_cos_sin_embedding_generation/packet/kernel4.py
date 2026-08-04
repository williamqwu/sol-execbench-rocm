import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

# Fused RoPE cos/sin embedding generation.
#
# Reference does:  emb = cat((freqs, freqs), -1); cos = (emb.cos()*s).bf16;
#                                                  sin = (emb.sin()*s).bf16
# The concat is pure duplication, so we never materialize `emb`: each program
# loads a [BR, D] tile of freqs once, computes cos/sin in fp32 (matching the
# reference's fp32 math before the bf16 round), and writes each result twice.
# That takes input traffic from 2x down to 1x.
#
# Stores use the `.wt` (write-through) cache modifier: the output is streamed
# and never re-read, so keeping it out of L2 measurably helps the large shapes.


@triton.jit
def _emb_masked(fp, cp, sp, n_rows, scale,
                D: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, D)
    m = r[:, None] < n_rows
    x = tl.load(fp + r[:, None] * D + c[None, :], mask=m, other=0.0)
    co = (tl.cos(x) * scale).to(tl.bfloat16)
    si = (tl.sin(x) * scale).to(tl.bfloat16)
    o = r[:, None] * (2 * D) + c[None, :]
    tl.store(cp + o, co, mask=m, cache_modifier=".wt")
    tl.store(cp + o + D, co, mask=m, cache_modifier=".wt")
    tl.store(sp + o, si, mask=m, cache_modifier=".wt")
    tl.store(sp + o + D, si, mask=m, cache_modifier=".wt")


@triton.jit
def _emb_exact(fp, cp, sp, n_rows, scale,
               D: tl.constexpr, BR: tl.constexpr):
    # BR divides n_rows exactly -> no bounds math, no masks
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, D)
    x = tl.load(fp + r[:, None] * D + c[None, :])
    co = (tl.cos(x) * scale).to(tl.bfloat16)
    si = (tl.sin(x) * scale).to(tl.bfloat16)
    o = r[:, None] * (2 * D) + c[None, :]
    tl.store(cp + o, co, cache_modifier=".wt")
    tl.store(cp + o + D, co, cache_modifier=".wt")
    tl.store(sp + o, si, cache_modifier=".wt")
    tl.store(sp + o + D, si, cache_modifier=".wt")


_bf16 = torch.bfloat16
_empty = torch.empty
# Triton's own raw stream getter; ~25x cheaper than torch.cuda.current_stream()
# and reads the same ambient (or graph-captured) stream.
_get_stream = _driver.active.get_current_stream

# (D, BR, exact) -> (raw_launch, function, packed_metadata)
_CACHE = {}


def _cfg(n_rows):
    """Tile height / wavefronts, from a sweep over the workload shapes."""
    if n_rows >= 16384:
        return 16, 4
    if n_rows >= 4096:
        return 8, 4
    if n_rows >= 1024:
        return 4, 4
    return 2, 4


def _build(D, BR, exact, warps, device):
    kern = _emb_exact if exact else _emb_masked
    di = _empty(BR, D, dtype=torch.float32, device=device)
    do = _empty(BR, 2 * D, dtype=_bf16, device=device)
    ck = kern[(1,)](di, do, do, BR, 1.0,
                    D=D, BR=BR, num_warps=warps, num_stages=1)
    ck._init_handles()
    ent = (ck.run, ck.function, ck.packed_metadata)
    _CACHE[(D, BR, exact)] = ent
    return ent


@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    if not freqs.is_contiguous():
        freqs = freqs.contiguous()

    shape = freqs.shape
    D = shape[-1]
    n_rows = freqs.numel() // D

    # one allocation for both outputs; unbind is the cheapest way to split it
    out = _empty((2,) + shape[:-1] + (2 * D,), dtype=_bf16, device=freqs.device)
    cos, sin = out.unbind(0)

    if n_rows == 0:
        return cos, sin

    BR, warps = _cfg(n_rows)
    exact = (n_rows % BR) == 0
    key = (D, BR, exact)
    ent = _CACHE.get(key)
    if ent is None:
        ent = _build(D, BR, exact, warps, freqs.device)
    run_, func, pm = ent

    run_((n_rows + BR - 1) // BR, 1, 1,
         _get_stream(freqs.device.index), func, pm, None, None, None,
         freqs, cos, sin, n_rows, attention_scaling, D, BR)
    return cos, sin
