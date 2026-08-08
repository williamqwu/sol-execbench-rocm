import torch
import torch.nn.functional as F
import triton
import triton.language as tl

SOFTCAP = 30.0


@triton.jit
def _tanh(x):
    # tanh(x) = sign(x) * (1 - exp(-2|x|)) / (1 + exp(-2|x|))
    a = tl.abs(x)
    t = tl.exp(-2.0 * a)
    r = (1.0 - t) / (1.0 + t)
    return tl.where(x < 0, -r, r)


@triton.jit
def _softcap(x, CAP: tl.constexpr):
    # Replicate the reference's bf16 round-trips exactly:
    #   scaled = (x / 30).to(bf16); clamped = tanh(scaled).to(bf16)
    #   softcapped = (clamped * 30).to(bf16)
    s = (x / CAP).to(tl.bfloat16).to(tl.float32)
    c = _tanh(s).to(tl.bfloat16).to(tl.float32)
    return (c * CAP).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _softcap_softmax_1pass(X, Y, n_rows, n_cols,
                           CAP: tl.constexpr,
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
    sc = tl.where(mask, _softcap(x, CAP), float('-inf'))

    m = tl.max(sc, axis=1)
    e = tl.exp(sc - m[:, None])
    z = tl.sum(e, axis=1)
    y = e / z[:, None]

    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _softcap_softmax_2pass(X, Y, n_rows, n_cols,
                           CAP: tl.constexpr,
                           BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = row * n_cols

    m = float('-inf')
    z = 0.0
    for start in range(0, n_cols, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        sc = tl.where(mask, _softcap(x, CAP), float('-inf'))
        m_new = tl.maximum(m, tl.max(sc, axis=0))
        z = z * tl.exp(m - m_new) + tl.sum(tl.exp(sc - m_new), axis=0)
        m = m_new

    inv = 1.0 / z
    for start in range(0, n_cols, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.exp(_softcap(x, CAP) - m) * inv
        tl.store(Y + base + cols, y.to(Y.dtype.element_ty), mask=mask)


_cfg_cache = {}


def _config(n_cols):
    cfg = _cfg_cache.get(n_cols)
    if cfg is not None:
        return cfg
    block_n = triton.next_power_of_2(n_cols)
    target = 4096
    rows = max(1, target // block_n)
    if block_n <= 256:
        num_warps = 4
    elif block_n <= 1024:
        num_warps = 4
    else:
        num_warps = 8
    cfg = (block_n, rows, num_warps)
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
        grid = (triton.cdiv(n_rows, rows),)
        _softcap_softmax_1pass[grid](
            x, out, n_rows, n_cols,
            CAP=SOFTCAP, BLOCK_N=block_n, ROWS=rows,
            EVEN=(block_n == n_cols),
            num_warps=num_warps, num_stages=1,
        )
    else:
        _softcap_softmax_2pass[(n_rows,)](
            x, out, n_rows, n_cols,
            CAP=SOFTCAP, BLOCK_N=2048,
            num_warps=8, num_stages=1,
        )
    return out
