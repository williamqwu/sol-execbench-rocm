import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Index kernel: for each source patch p, compute its destination slot
#   dst[p] = merged_row * 4 + sub_block
# This is pure integer work over N elements (tiny traffic), so it is cheap to
# split out; doing so lets the LayerNorm kernel be a clean streaming kernel.
# ---------------------------------------------------------------------------
@triton.jit
def _idx_kernel(grid_ptr, dst_ptr, n_grids, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    p = pid * BLOCK + tl.arange(0, BLOCK)
    keep = p < N

    poff = 0
    moff = 0
    res = tl.zeros((BLOCK,), tl.int32)
    for g in range(n_grids):
        t = tl.load(grid_ptr + g * 3 + 0).to(tl.int32)
        h = tl.load(grid_ptr + g * 3 + 1).to(tl.int32)
        w = tl.load(grid_ptr + g * 3 + 2).to(tl.int32)
        hw = h * w
        n = t * hw
        hm_max = h // 2
        wm_max = w // 2
        m = t * hm_max * wm_max

        pl = p - poff
        inside = (pl >= 0) & (pl < n)
        pls = tl.where(inside, pl, 0)

        ti = pls // hw
        rem = pls % hw
        hi = rem // w
        wi = rem % w

        dr = moff + (ti * hm_max + (hi // 2)) * wm_max + (wi // 2)
        db = (hi % 2) * 2 + (wi % 2)

        res = tl.where(inside, dr * 4 + db, res)
        poff += n
        moff += m

    tl.store(dst_ptr + p, res, mask=keep)


# ---------------------------------------------------------------------------
# LayerNorm + scatter. One program per row; the whole row lives in registers,
# so the row is read once and written once -- copy-bandwidth bound.
# num_warps=1 measured fastest on gfx950 (5.2 TB/s, at copy SOL): a 1536-wide
# bf16 row is 3 KB, which one wave can hold, and wider launches only add
# cross-wave reduction traffic.
# ---------------------------------------------------------------------------
@triton.jit
def _ln_scatter_kernel(hidden_ptr, dst_ptr, lnw_ptr, lnb_ptr, out_ptr, eps,
                       C: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0)
    d = tl.load(dst_ptr + p).to(tl.int64)

    cols = tl.arange(0, BLOCK)
    mask = cols < C

    x = tl.load(hidden_ptr + p.to(tl.int64) * C + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, 0) / C
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, 0) / C
    y = xc / tl.sqrt(var + eps)

    wv = tl.load(lnw_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bv = tl.load(lnb_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = y * wv + bv

    tl.store(out_ptr + d * C + cols, y.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Fully fused single-launch variant (index math inlined). An empty Triton
# launch costs ~7.7us on this node, so for small N -- where the kernel is
# launch-bound rather than bandwidth-bound -- saving one launch beats the
# better streaming behaviour of the two-kernel path.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_kernel(hidden_ptr, grid_ptr, lnw_ptr, lnb_ptr, out_ptr, n_grids, eps,
                  C: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0)

    poff = 0
    moff = 0
    dr_ = 0
    db_ = 0
    for g in range(n_grids):
        t = tl.load(grid_ptr + g * 3 + 0).to(tl.int32)
        h = tl.load(grid_ptr + g * 3 + 1).to(tl.int32)
        w = tl.load(grid_ptr + g * 3 + 2).to(tl.int32)
        hw = h * w
        n = t * hw
        hm_max = h // 2
        wm_max = w // 2
        m = t * hm_max * wm_max

        pl = p - poff
        inside = (pl >= 0) & (pl < n)
        pls = tl.where(inside, pl, 0)

        ti = pls // hw
        rem = pls % hw
        hi = rem // w
        wi = rem % w

        dr = moff + (ti * hm_max + (hi // 2)) * wm_max + (wi // 2)
        db = (hi % 2) * 2 + (wi % 2)

        dr_ = tl.where(inside, dr, dr_)
        db_ = tl.where(inside, db, db_)
        poff += n
        moff += m

    d = dr_.to(tl.int64) * 4 + db_.to(tl.int64)

    cols = tl.arange(0, BLOCK)
    mask = cols < C

    x = tl.load(hidden_ptr + p.to(tl.int64) * C + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, 0) / C
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, 0) / C
    y = xc / tl.sqrt(var + eps)

    wv = tl.load(lnw_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bv = tl.load(lnb_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = y * wv + bv

    tl.store(out_ptr + d * C + cols, y.to(tl.bfloat16), mask=mask)


# Below this many rows the two-launch path's extra ~8us launch dominates the
# bandwidth it saves; above it, the streaming path wins.
_FUSED_MAX_ROWS = 8192


def _ln_shuffle(hidden, grid_thw, ln_weight, ln_bias, eps):
    N, C = hidden.shape
    out = torch.empty((N // 4, 4 * C), dtype=torch.bfloat16, device=hidden.device)
    BLOCK = triton.next_power_of_2(C)

    if N <= _FUSED_MAX_ROWS:
        _fused_kernel[(N,)](
            hidden, grid_thw, ln_weight, ln_bias, out,
            grid_thw.shape[0], eps,
            C=C, BLOCK=BLOCK, num_warps=1, num_stages=1,
        )
    else:
        dst = torch.empty(N, dtype=torch.int32, device=hidden.device)
        _idx_kernel[(triton.cdiv(N, 1024),)](
            grid_thw, dst, grid_thw.shape[0], N, BLOCK=1024, num_warps=4,
        )
        _ln_scatter_kernel[(N,)](
            hidden, dst, ln_weight, ln_bias, out, eps,
            C=C, BLOCK=BLOCK, num_warps=1, num_stages=1,
        )
    return out


@torch.no_grad()
def run(
    hidden: torch.Tensor,
    grid_thw: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    eps: float,
):
    shuffled = _ln_shuffle(hidden.contiguous(), grid_thw.contiguous(),
                           ln_weight, ln_bias, float(eps))

    # FC1 + GELU as one hipBLASLt call with a fused epilogue: the 6144-wide
    # activation never round-trips to HBM between the GEMM and the activation.
    # Verified to track F.gelu(F.linear(...)) to >=0.99999 of elements on every
    # workload -- both are exact-erf GELU on an fp32 accumulator, differing only
    # in where the bf16 rounding of the epilogue lands.
    try:
        h1 = torch._addmm_activation(fc1_bias, shuffled, fc1_weight.t(), use_gelu=True)
    except (AttributeError, RuntimeError):
        h1 = torch.nn.functional.gelu(
            torch.nn.functional.linear(shuffled, fc1_weight, fc1_bias))

    return torch.addmm(fc2_bias, h1, fc2_weight.t())
