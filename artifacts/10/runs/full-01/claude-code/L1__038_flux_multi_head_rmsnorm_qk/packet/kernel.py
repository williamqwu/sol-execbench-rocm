import torch
import triton
import triton.language as tl


@triton.jit
def _rms_qk(Q, K, WQ, WK, OQ, OK, n_rows, eps,
            HEADS: tl.constexpr, DD: tl.constexpr, ROWS: tl.constexpr):
    pid = tl.program_id(0)
    which = tl.program_id(1)

    rows = pid * ROWS + tl.arange(0, ROWS)
    mask = rows < n_rows
    cols = tl.arange(0, DD)
    offs = rows[:, None] * DD + cols[None, :]

    if which == 0:
        xp = Q
        wp = WQ
        op = OQ
    else:
        xp = K
        wp = WK
        op = OK

    x = tl.load(xp + offs, mask=mask[:, None], other=0.0)

    # matches reference: float32 mean of squares over head_dim, then rsqrt
    var = tl.sum(x * x, axis=1) * (1.0 / DD)
    scale = tl.math.rsqrt(var + eps)

    h = rows % HEADS
    w = tl.load(wp + h[:, None] * DD + cols[None, :], mask=mask[:, None], other=0.0)

    tl.store(op + offs, (x * scale[:, None]) * w, mask=mask[:, None])


_ROWS = 4
_cache = {}

try:
    _get_stream = torch._C._cuda_getCurrentRawStream
except AttributeError:  # pragma: no cover
    _get_stream = None


def run(query: torch.Tensor, key: torch.Tensor, weight_q: torch.Tensor,
        weight_k: torch.Tensor, eps: float):
    q = query if query.is_contiguous() else query.contiguous()
    k = key if key.is_contiguous() else key.contiguous()
    wq = weight_q if weight_q.is_contiguous() else weight_q.contiguous()
    wk = weight_k if weight_k.is_contiguous() else weight_k.contiguous()

    heads = wq.shape[0]
    d = wq.shape[-1]
    n = q.numel() // d

    oq = torch.empty_like(q)
    ok = torch.empty_like(k)
    if n == 0:
        return oq.view(query.shape), ok.view(key.shape)

    gx = (n + _ROWS - 1) // _ROWS

    # Cache on everything Triton specializes over, so a cached launcher is only
    # ever reused for a kernel compiled under identical specializations.
    # Alignment is tracked per-pointer, not OR-reduced: Triton emits alignment
    # hints per argument, so a combined flag could alias two different
    # per-pointer patterns onto one entry and reuse a kernel that assumes an
    # alignment a later tensor does not have.
    ckey = (heads, d, n % 16 == 0, n == 1, q.device.index,
            q.data_ptr() % 16 == 0, k.data_ptr() % 16 == 0,
            wq.data_ptr() % 16 == 0, wk.data_ptr() % 16 == 0,
            oq.data_ptr() % 16 == 0, ok.data_ptr() % 16 == 0)
    ck = _cache.get(ckey)

    if ck is None or _get_stream is None:
        compiled = _rms_qk[(gx, 2)](
            q, k, wq, wk, oq, ok, n, eps,
            HEADS=heads, DD=d, ROWS=_ROWS,
            num_warps=1, num_stages=1,
        )
        if _get_stream is not None:
            _cache[ckey] = (compiled.run, compiled.function,
                            compiled.packed_metadata)
        return oq.view(query.shape), ok.view(key.shape)

    launch, fn, pmeta = ck
    launch(gx, 2, 1, _get_stream(q.device.index), fn, pmeta, None, None, None,
           q, k, wq, wk, oq, ok, n, eps, heads, d, _ROWS)

    return oq.view(query.shape), ok.view(key.shape)
