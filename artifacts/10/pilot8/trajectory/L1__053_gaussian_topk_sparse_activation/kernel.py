import torch
import triton
import triton.language as tl
import numpy as np


# ---------------------------------------------------------------------------
# Inverse normal CDF (Abramowitz & Stegun 26.2.23), evaluated on the host in
# float32 so the arithmetic matches the reference's fp32 GPU evaluation.
# ---------------------------------------------------------------------------
_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]

_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW

_f32 = np.float32


def _ndtri_scalar(p: float) -> float:
    """float32 evaluation of the same rational approximation as the reference."""
    p = _f32(p)
    if p < _f32(_P_LOW):
        q = np.sqrt(_f32(-2.0) * np.log(p), dtype=_f32)
        num = _f32(_C[0])
        for c in _C[1:]:
            num = _f32(num * q + _f32(c))
        den = _f32(_D[0])
        for d in _D[1:]:
            den = _f32(den * q + _f32(d))
        den = _f32(den * q + _f32(1.0))
        return float(_f32(num / den))
    if p <= _f32(_P_HIGH):
        q = _f32(p - _f32(0.5))
        r = _f32(q * q)
        num = _f32(_A[0])
        for a in _A[1:]:
            num = _f32(num * r + _f32(a))
        num = _f32(num * q)
        den = _f32(_B[0])
        for b in _B[1:]:
            den = _f32(den * r + _f32(b))
        den = _f32(den * r + _f32(1.0))
        return float(_f32(num / den))
    q = np.sqrt(_f32(-2.0) * np.log(_f32(_f32(1.0) - p)), dtype=_f32)
    num = _f32(_C[0])
    for c in _C[1:]:
        num = _f32(num * q + _f32(c))
    den = _f32(_D[0])
    for d in _D[1:]:
        den = _f32(den * q + _f32(d))
    den = _f32(den * q + _f32(1.0))
    return float(_f32(-(num / den)))


_ndtri_cache = {}


def _ndtri_cached(p: float) -> float:
    v = _ndtri_cache.get(p)
    if v is None:
        v = _ndtri_scalar(p)
        _ndtri_cache[p] = v
    return v


# ---------------------------------------------------------------------------
# Fused kernel: one program per row, whole row held in registers.
# ---------------------------------------------------------------------------
@triton.jit
def _gauss_topk_row(X, Y, mult, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    base = row * N
    if BLOCK == N:
        x = tl.load(X + base + cols).to(tl.float32)
        mean = tl.sum(x, axis=0) * (1.0 / N)
        d = x - mean
        var = tl.sum(d * d, axis=0) * (1.0 / N)
        thr = mean + tl.sqrt(var) * mult
        y = tl.maximum(x - thr, 0.0)
        tl.store(Y + base + cols, y.to(tl.bfloat16))
    else:
        mask = cols < N
        x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) * (1.0 / N)
        d = tl.where(mask, x - mean, 0.0)
        var = tl.sum(d * d, axis=0) * (1.0 / N)
        thr = mean + tl.sqrt(var) * mult
        y = tl.maximum(x - thr, 0.0)
        tl.store(Y + base + cols, y.to(tl.bfloat16), mask=mask)


def _cfg(N):
    block = triton.next_power_of_2(N)
    if block <= 4096:
        return block, 4
    if block <= 8192:
        return block, 8
    return block, 16


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    if target_sparsity == 0.0:
        return inputs

    x = inputs if inputs.is_contiguous() else inputs.contiguous()
    N = x.shape[-1]
    rows = x.numel() // N

    out = torch.empty_like(x)
    mult = _ndtri_cached(float(target_sparsity))

    block, num_warps = _cfg(N)
    _gauss_topk_row[(rows,)](
        x, out, mult, N=N, BLOCK=block,
        num_warps=num_warps, num_stages=1,
    )
    return out
