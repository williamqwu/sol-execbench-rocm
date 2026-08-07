import torch
import torch.nn.functional as F
import math
import triton
import triton.language as tl


def _ndtri_scalar(p: float) -> float:
    """Inverse of the standard normal CDF (quantile function), pure Python."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")

    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
            (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / (
            ((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
            (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)


@triton.jit
def _stats_kernel(
    x_ptr,
    mean_ptr,
    sqmean_ptr,
    n_cols,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * n_cols
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.sum(x, axis=0) / n_cols
    sq = tl.sum(x * x, axis=0) / n_cols
    tl.store(mean_ptr + pid, m)
    tl.store(sqmean_ptr + pid, sq)


@triton.jit
def _relu_cast_kernel(
    x_ptr,
    thresh_ptr,
    out_ptr,
    n_cols,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * n_cols
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    thresh = tl.load(thresh_ptr + pid)
    y = x.to(tl.float32) - thresh
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + base + offs, y.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    if target_sparsity == 0.0:
        return inputs

    std_multiplier = _ndtri_scalar(target_sparsity)
    orig_shape = inputs.shape
    n_cols = orig_shape[-1]
    x2d = inputs.reshape(-1, n_cols).contiguous()
    nrows = x2d.shape[0]

    mean_t = torch.empty(nrows, dtype=torch.float32, device=inputs.device)
    sqmean_t = torch.empty(nrows, dtype=torch.float32, device=inputs.device)

    BLOCK = triton.next_power_of_2(n_cols)
    BLOCK = max(BLOCK, 64)
    num_warps = 16 if BLOCK >= 8192 else (8 if BLOCK >= 2048 else 4)

    _stats_kernel[(nrows,)](
        x2d, mean_t, sqmean_t, n_cols,
        BLOCK=BLOCK, num_warps=num_warps,
    )

    var_t = (sqmean_t - mean_t * mean_t).clamp_(min=0.0)
    std_t = torch.sqrt_(var_t)
    thresh_t = mean_t + std_t * std_multiplier

    out2d = torch.empty_like(x2d)
    R_BLOCK = 4096
    _relu_cast_kernel[(nrows,)](
        x2d, thresh_t, out2d, n_cols,
        BLOCK=R_BLOCK, num_warps=8,
    )
    return out2d.view(orig_shape)
