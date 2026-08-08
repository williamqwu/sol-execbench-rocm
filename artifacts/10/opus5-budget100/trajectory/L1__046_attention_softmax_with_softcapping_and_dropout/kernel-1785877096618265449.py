import torch
import torch.nn.functional as F
import triton
import triton.language as tl

SOFTCAP = 30.0
LOG2E = 1.4426950408889634


@triton.jit
def _rcp(x):
    """Hardware reciprocal (~1 ulp). Triton's `/` and `fdiv` both expand to the
    full IEEE div_scale/div_fmas/div_fixup sequence (~5 VALU ops); this kernel
    is VALU-bound and the extra accuracy is discarded by the bf16 rounding that
    immediately follows.
    """
    return tl.inline_asm_elementwise("v_rcp_f32 $0, $1", "=v,v", [x],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _softcap_exp(x, CAP: tl.constexpr, INV_CAP: tl.constexpr,
                 NEG2_LOG2E: tl.constexpr, LOG2E: tl.constexpr):
    """exp(softcap(x)) where softcap(x) = tanh(x/30)*30, mirroring the
    reference's bf16 round-trips between each step.

    tanh(a) = 2/(1 + exp(-2a)) - 1, valid over the whole line: as a -> -inf the
    exp overflows to +inf and 2/inf - 1 = -1, which is the right limit, so no
    sign/abs branch is needed. Clamping first keeps everything finite.
    """
    a = (tl.maximum(x, -600.0) * INV_CAP).to(tl.bfloat16).to(tl.float32)
    t = tl.math.exp2(a * NEG2_LOG2E)
    r = (2.0 * _rcp(1.0 + t) - 1.0).to(tl.bfloat16).to(tl.float32)
    # softcapped = (r * 30).to(bf16); exp(s) == exp2(s * log2e)
    s = (r * CAP).to(tl.bfloat16).to(tl.float32)
    return tl.math.exp2(s * LOG2E)


@triton.jit
def _softcap_softmax_1pass(X, Y, n_rows, n_cols,
                           CAP: tl.constexpr,
                           INV_CAP: tl.constexpr,
                           NEG2_LOG2E: tl.constexpr,
                           LOG2E: tl.constexpr,
                           BLOCK_N: tl.constexpr,
                           ROWS: tl.constexpr,
                           EVEN: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_N)
    off = rows[:, None].to(tl.int64) * n_cols + cols[None, :]

    rmask = rows[:, None] < n_rows
    if EVEN:
        mask = tl.broadcast_to(rmask, (ROWS, BLOCK_N))
    else:
        mask = rmask & (cols[None, :] < n_cols)

    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    # softcap bounds the logits to [-30, 30], so exp() cannot overflow fp32
    # (exp(30) ~ 1e13) and the usual max-subtraction pass is unnecessary.
    f = tl.where(mask, _softcap_exp(x, CAP, INV_CAP, NEG2_LOG2E, LOG2E), 0.0)
    z = tl.sum(f, axis=1)
    y = f * _rcp(z)[:, None]

    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _softcap_softmax_2pass(X, Y, n_rows, n_cols,
                           CAP: tl.constexpr,
                           INV_CAP: tl.constexpr,
                           NEG2_LOG2E: tl.constexpr,
                           LOG2E: tl.constexpr,
                           BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = row * n_cols

    z = 0.0
    for start in range(0, n_cols, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        z += tl.sum(tl.where(mask, _softcap_exp(x, CAP, INV_CAP, NEG2_LOG2E, LOG2E), 0.0), axis=0)

    inv = _rcp(z)
    for start in range(0, n_cols, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        y = _softcap_exp(x, CAP, INV_CAP, NEG2_LOG2E, LOG2E) * inv
        tl.store(Y + base + cols, y.to(Y.dtype.element_ty), mask=mask)


_cfg_cache = {}


def _config(n_cols, n_rows):
    key = (n_cols, n_rows)
    cfg = _cfg_cache.get(key)
    if cfg is not None:
        return cfg

    block_n = triton.next_power_of_2(n_cols)
    # Target 8 bf16 elements per lane, i.e. exactly one global_load_dwordx4 per
    # thread: measured best (or within noise of best) on every workload shape.
    # Keeping the tile at one wavefront where possible also keeps the row
    # reduction in cross-lane ops instead of an LDS round trip.
    lanes = max(1, min(1024, (block_n + 7) // 8))
    num_warps = max(1, min(8, lanes // 64))
    rows = max(1, (num_warps * 64 * 8) // block_n)

    # Do not launch fewer programs than we have CUs while rows can be reduced.
    while rows > 1 and (n_rows + rows - 1) // rows < 256:
        rows //= 2

    cfg = (block_n, rows, num_warps)
    _cfg_cache[key] = cfg
    return cfg


@torch.no_grad()
def run(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Apply Gemma3's softcapping transformation followed by softmax.

    Softcapping: tanh(logits / 30.0) * 30.0
    This clamps effective logit range to approximately [-30, +30]

    Args:
        attn_weights: Attention logits of shape (batch_size, num_heads, seq_len_q, seq_len_k)

    Returns:
        Normalized attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    if not attn_weights.is_cuda:
        softcapped = torch.tanh(attn_weights / SOFTCAP) * SOFTCAP
        return F.softmax(softcapped, dim=-1, dtype=torch.float32).to(attn_weights.dtype)

    x = attn_weights.contiguous()
    n_cols = x.shape[-1]
    n_rows = (x.numel() // n_cols) if n_cols else 0
    out = torch.empty_like(x)
    if n_rows == 0 or n_cols == 0:
        return out

    if n_cols <= 8192:
        block_n, rows, num_warps = _config(n_cols, n_rows)
        _softcap_softmax_1pass[(triton.cdiv(n_rows, rows),)](
            x, out, n_rows, n_cols,
            CAP=SOFTCAP, INV_CAP=1.0 / SOFTCAP,
            NEG2_LOG2E=-2.0 * LOG2E, LOG2E=LOG2E,
            BLOCK_N=block_n, ROWS=rows, EVEN=(block_n == n_cols),
            num_warps=num_warps, num_stages=1,
        )
    else:
        _softcap_softmax_2pass[(n_rows,)](
            x, out, n_rows, n_cols,
            CAP=SOFTCAP, INV_CAP=1.0 / SOFTCAP,
            NEG2_LOG2E=-2.0 * LOG2E, LOG2E=LOG2E,
            BLOCK_N=2048, num_warps=8, num_stages=1,
        )
    return out
