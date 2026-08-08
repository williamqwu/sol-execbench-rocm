import torch
import torch.nn.functional as F
import triton
import triton.language as tl

SOFTCAP = 30.0
LOG2E = 1.4426950408889634


@triton.jit
def _softcap_exp(x, NEG2_LOG2E: tl.constexpr, CAP_LOG2E: tl.constexpr):
    """exp(softcap(x)) where softcap(x) = tanh(x/30)*30, mirroring the
    reference's bf16 round-trips between each step.

    tanh(a) = sign(a) * (1 - exp(-2|a|)) / (1 + exp(-2|a|))
    """
    a = tl.abs(x) * (1.0 / SOFTCAP)
    a = a.to(tl.bfloat16).to(tl.float32)
    t = tl.math.exp2(a * NEG2_LOG2E)
    r = tl.fdiv(1.0 - t, 1.0 + t, ieee_rounding=False)
    r = tl.where(x < 0, -r, r).to(tl.bfloat16).to(tl.float32)
    # softcapped = (r * 30).to(bf16); exp(s) == exp2(s * log2e)
    s = (r * SOFTCAP).to(tl.bfloat16).to(tl.float32)
    return tl.math.exp2(s * LOG2E)


@triton.jit
def _softcap_softmax_1pass(X, Y, n_rows, n_cols,
                           NEG2_LOG2E: tl.constexpr,
                           CAP_LOG2E: tl.constexpr,
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
    f = tl.where(mask, _softcap_exp(x, NEG2_LOG2E, CAP_LOG2E), 0.0)
    z = tl.sum(f, axis=1)
    y = f * tl.fdiv(1.0, z, ieee_rounding=False)[:, None]

    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _softcap_softmax_2pass(X, Y, n_rows, n_cols,
                           NEG2_LOG2E: tl.constexpr,
                           CAP_LOG2E: tl.constexpr,
                           BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = row * n_cols

    z = 0.0
    for start in range(0, n_cols, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        z += tl.sum(tl.where(mask, _softcap_exp(x, NEG2_LOG2E, CAP_LOG2E), 0.0), axis=0)

    inv = 1.0 / z
    for start in range(0, n_cols, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        y = _softcap_exp(x, NEG2_LOG2E, CAP_LOG2E) * inv
        tl.store(Y + base + cols, y.to(Y.dtype.element_ty), mask=mask)


_cfg_cache = {}


def _config(n_cols):
    cfg = _cfg_cache.get(n_cols)
    if cfg is not None:
        return cfg
    block_n = triton.next_power_of_2(n_cols)
    # One wavefront per program: the row reduction stays in registers /
    # cross-lane ops with no LDS round trip, and the fine grain spreads
    # work evenly over 256 CUs.
    rows = max(1, 512 // block_n)
    cfg = (block_n, rows, 1)
    _cfg_cache[n_cols] = cfg
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
        block_n, rows, num_warps = _config(n_cols)
        _softcap_softmax_1pass[(triton.cdiv(n_rows, rows),)](
            x, out, n_rows, n_cols,
            NEG2_LOG2E=-2.0 * LOG2E, CAP_LOG2E=SOFTCAP * LOG2E,
            BLOCK_N=block_n, ROWS=rows, EVEN=(block_n == n_cols),
            num_warps=num_warps, num_stages=1,
        )
    else:
        _softcap_softmax_2pass[(n_rows,)](
            x, out, n_rows, n_cols,
            NEG2_LOG2E=-2.0 * LOG2E, CAP_LOG2E=SOFTCAP * LOG2E,
            BLOCK_N=2048, num_warps=8, num_stages=1,
        )
    return out
