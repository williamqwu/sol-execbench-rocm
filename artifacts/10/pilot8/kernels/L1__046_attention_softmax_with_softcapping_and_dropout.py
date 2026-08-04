import torch
import triton
import triton.language as tl

INV_SC = tl.constexpr(1.0 / 30.0)
SC = tl.constexpr(30.0)
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _rbf(x):
    """Round an fp32 value to bfloat16 precision and back."""
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _tanh(s):
    """tanh via the hardware exp2; well under bf16 precision.

    |s| < 2^-5 : tanh(s) == s to far better than bf16, and this branch dodges
                 the catastrophic cancellation in (1-e)/(1+e) near 0.
    """
    a = tl.abs(s)
    e = tl.exp2((-2.0 * LOG2E) * a)
    t = (1.0 - e) / (1.0 + e)
    t = tl.where(a < 0.03125, a, t)
    return tl.where(s < 0, -t, t)


@triton.jit
def _softcap(xb):
    """Bit-exact copy of the reference chain, which stays bf16 at every step:
    (x / 30.0) -> tanh -> (* 30.0), each rounded to bf16."""
    s = _rbf(xb.to(tl.float32) * INV_SC)
    t = _rbf(_tanh(s))
    return _rbf(t * SC)


@triton.jit
def _k_rows(X, Y, n_rows, N, BLOCK_N: tl.constexpr, ROWS: tl.constexpr,
            MAXSUB: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_N)
    rm = rows < n_rows
    m = rm[:, None] if EVEN else (rm[:, None] & (cols[None, :] < N))
    off = rows[:, None].to(tl.int64) * N + cols[None, :]

    u = _softcap(tl.load(X + off, mask=m, other=0.0))
    if MAXSUB:
        u = tl.where(m, u, -float('inf'))
        u = u - tl.max(u, 1)[:, None]
    e = tl.exp2(u * LOG2E)
    e = tl.where(m, e, 0.0)
    y = e / tl.sum(e, 1)[:, None]
    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=m)


@triton.jit
def _k_split(X, Y, n_rows, N, BLOCK_N: tl.constexpr, MAXSUB: tl.constexpr):
    """One program per row, streaming in BLOCK_N chunks (large N)."""
    row = tl.program_id(0)
    base = row.to(tl.int64) * N
    mx = 0.0
    if MAXSUB:
        mx = -float('inf')
        for st in range(0, N, BLOCK_N):
            c = st + tl.arange(0, BLOCK_N)
            u = _softcap(tl.load(X + base + c, mask=c < N, other=0.0))
            u = tl.where(c < N, u, -float('inf'))
            mx = tl.maximum(mx, tl.max(u, 0))
    s = 0.0
    for st in range(0, N, BLOCK_N):
        c = st + tl.arange(0, BLOCK_N)
        u = _softcap(tl.load(X + base + c, mask=c < N, other=0.0))
        e = tl.exp2((u - mx) * LOG2E)
        s += tl.sum(tl.where(c < N, e, 0.0), 0)
    inv = 1.0 / s
    for st in range(0, N, BLOCK_N):
        c = st + tl.arange(0, BLOCK_N)
        u = _softcap(tl.load(X + base + c, mask=c < N, other=0.0))
        e = tl.exp2((u - mx) * LOG2E) * inv
        tl.store(Y + base + c, e.to(Y.dtype.element_ty), mask=c < N)


@triton.jit
def _k_dbg(X, Y, n, BLOCK: tl.constexpr):
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + c, _softcap(tl.load(X + c, mask=c < n, other=0.0)).to(Y.dtype.element_ty),
             mask=c < n)


MAXSUB = False
_cache = {}


def _plan(N, n_rows):
    key = (N, n_rows)
    p = _cache.get(key)
    if p is None:
        bn = triton.next_power_of_2(N)
        rows = max(1, min(16, 4096 // bn))
        while rows > 1 and triton.cdiv(n_rows, rows) < 1024:
            rows //= 2
        nw = 4 if bn <= 1024 else 8
        p = (bn, rows, nw)
        _cache[key] = p
    return p


@torch.no_grad()
def run(attn_weights: torch.Tensor) -> torch.Tensor:
    x = attn_weights if attn_weights.is_contiguous() else attn_weights.contiguous()
    N = x.shape[-1]
    n_rows = x.numel() // N
    y = torch.empty_like(x)
    if N <= 16384:
        bn, rows, nw = _plan(N, n_rows)
        _k_rows[(triton.cdiv(n_rows, rows),)](
            x, y, n_rows, N, BLOCK_N=bn, ROWS=rows, MAXSUB=MAXSUB,
            EVEN=(N == bn), num_warps=nw, num_stages=1)
    else:
        _k_split[(n_rows,)](x, y, n_rows, N, BLOCK_N=4096, MAXSUB=MAXSUB,
                            num_warps=8, num_stages=1)
    return y
