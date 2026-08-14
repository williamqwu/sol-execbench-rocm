import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# L1/046 -- Gemma-3 attention logit softcapping + softmax.
#
#   scaled     = attn_weights / 30.0        (bfloat16)
#   clamped    = tanh(scaled)               (bfloat16)
#   softcapped = clamped * 30.0             (bfloat16)
#   out        = softmax(softcapped, dim=-1, dtype=float32).to(bfloat16)
#
# The reference performs each softcap step in bfloat16, so the intermediate
# roundings are part of the spec and are reproduced exactly below (bf16 round
# after each of the three steps).  The softmax itself is done in float32.
#
# Two structural optimisations:
#
#  1. No max-subtraction pass.  tanh() bounds `softcapped` to [-30, +30]
#     *by construction*, so exp(c - 30) can never overflow (argument <= 0) and
#     can never flush to zero (argument >= -60, i.e. exp2 arg >= -86.6, still
#     comfortably inside float32 normals ~1e-26).  This removes an entire
#     reduction over the row -- worth ~1.7x on the large shapes.
#  2. exp2 instead of exp -- maps to the native v_exp_f32 instruction.
#
# Everything is fused into a single pass: one read of the input, one write of
# the output, no intermediate tensors.
# ---------------------------------------------------------------------------

LOG2E: tl.constexpr = tl.constexpr(1.4426950408889634)


@triton.jit
def _softcap_softmax(
    X, Y,
    n_rows, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = tl.arange(0, BLOCK_N)

    offs = rm[:, None].to(tl.int64) * N + cn[None, :]

    if EVEN_M:
        if EVEN_N:
            x = tl.load(X + offs)
        else:
            x = tl.load(X + offs, mask=cn[None, :] < N, other=0.0)
    else:
        row_ok = rm < n_rows
        if EVEN_N:
            x = tl.load(X + offs, mask=row_ok[:, None], other=0.0)
        else:
            x = tl.load(X + offs, mask=row_ok[:, None] & (cn[None, :] < N), other=0.0)

    # --- softcap, reproducing the reference's bf16 intermediate rounding -----
    xf = x.to(tl.float32)
    s = (xf * (1.0 / 30.0)).to(tl.bfloat16).to(tl.float32)
    t = libdevice.tanh(s).to(tl.bfloat16).to(tl.float32)
    c = (t * 30.0).to(tl.bfloat16).to(tl.float32)

    # --- softmax (float32), no max pass: c is bounded above by +30 -----------
    if EVEN_N:
        e = tl.exp2((c - 30.0) * LOG2E)
    else:
        # out-of-range columns must not contribute to the sum
        e = tl.where(cn[None, :] < N, tl.exp2((c - 30.0) * LOG2E), 0.0)

    d = tl.sum(e, 1)
    y = e * (1.0 / d)[:, None]

    yb = y.to(tl.bfloat16)
    if EVEN_M:
        if EVEN_N:
            tl.store(Y + offs, yb)
        else:
            tl.store(Y + offs, yb, mask=cn[None, :] < N)
    else:
        row_ok2 = rm < n_rows
        if EVEN_N:
            tl.store(Y + offs, yb, mask=row_ok2[:, None])
        else:
            tl.store(Y + offs, yb, mask=row_ok2[:, None] & (cn[None, :] < N))


# ---------------------------------------------------------------------------
# Launch configuration.
# ---------------------------------------------------------------------------

def _pick(N, n_rows):
    """Return (BLOCK_N, BLOCK_M, num_warps) for a row length / row count."""
    BLOCK_N = max(triton.next_power_of_2(N), 16)

    if BLOCK_N >= 2048:
        BM, nw = 2, 4
    elif BLOCK_N >= 1024:
        BM, nw = 2, 4
    elif BLOCK_N >= 512:
        BM, nw = 8, 8
    elif BLOCK_N >= 256:
        BM, nw = 8, 4
    elif BLOCK_N >= 128:
        BM, nw = 8, 4
    else:
        BM, nw = 16, 4

    # Make sure we still fill the machine when there are few rows.
    CU = 256
    while BM > 1 and (n_rows + BM - 1) // BM < CU:
        BM //= 2
    return BLOCK_N, BM, nw


# Fast raw-stream accessor: torch.cuda.current_stream().cuda_stream costs
# ~2.5us of Python per call; this C shortcut costs ~0.06us and returns the
# identical value.
try:
    from torch._C import _cuda_getCurrentRawStream as _raw_stream
except ImportError:  # pragma: no cover
    def _raw_stream(dev):
        return torch.cuda.current_stream().cuda_stream


# Cache of pre-compiled kernels -> bypasses the ~6us Triton JIT dispatch path.
_cache = {}


def _get(N, n_rows):
    BLOCK_N, BM, nw = _pick(N, n_rows)
    even_n = (BLOCK_N == N)
    even_m = (n_rows % BM == 0)
    key = (N, BLOCK_N, BM, nw, even_n, even_m)
    ent = _cache.get(key)
    if ent is None:
        ck = _softcap_softmax.warmup(
            torch.empty(1, device="cuda", dtype=torch.bfloat16),
            torch.empty(1, device="cuda", dtype=torch.bfloat16),
            n_rows, N,
            BLOCK_M=BM, BLOCK_N=BLOCK_N, EVEN_N=even_n, EVEN_M=even_m,
            num_warps=nw, num_stages=1,
            grid=(1,),
        )
        ck._init_handles()
        ent = (ck.run, ck.function, ck.packed_metadata, BM, BLOCK_N, even_n, even_m)
        _cache[key] = ent
    return ent


# Per-(N, n_rows) dispatch cache: everything that does not depend on the
# tensor's address is resolved once and reused.
_plan = {}


def _get_plan(N, n_rows):
    p = _plan.get((N, n_rows))
    if p is None:
        run_fn, fun, pm, BM, BLOCK_N, even_n, even_m = _get(N, n_rows)
        grid0 = (n_rows + BM - 1) // BM
        p = (run_fn, fun, pm, grid0, BM, BLOCK_N, even_n, even_m)
        _plan[(N, n_rows)] = p
    return p


def _torch_fallback(x):
    s = (x / 30.0)
    c = torch.tanh(s) * 30.0
    return torch.softmax(c.float(), dim=-1).to(x.dtype)


@torch.no_grad()
def run(attn_weights: torch.Tensor) -> torch.Tensor:
    x = attn_weights
    if not x.is_contiguous():
        x = x.contiguous()

    N = x.shape[-1]
    numel = x.numel()
    if numel == 0 or N == 0:
        return torch.empty_like(x)

    n_rows = numel // N

    # Single-tile strategy needs the whole row in registers; fall back for
    # pathologically long rows (never hit by this problem's workloads).
    if N > 16384:
        return _torch_fallback(x)

    out = torch.empty_like(x)

    run_fn, fun, pm, grid0, BM, BLOCK_N, even_n, even_m = _get_plan(N, n_rows)

    run_fn(grid0, 1, 1, _raw_stream(x.device.index), fun, pm, None, None, None,
           x, out, n_rows, N, BM, BLOCK_N, even_n, even_m)
    return out
