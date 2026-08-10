import math
import torch
import triton
import triton.language as tl


@triton.jit
def _fused(x, out, n_cols: tl.constexpr, BLOCK: tl.constexpr, Z: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    v = tl.load(x + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(v, axis=0) / n_cols
    centered = v - mean
    var = tl.sum(centered * centered, axis=0) / n_cols
    cutoff = mean + tl.sqrt(tl.maximum(var, 0.0)) * Z
    y = tl.maximum(v - cutoff, 0.0)
    tl.store(out + row * n_cols + cols, y, mask=mask)


def _ndtri_scalar(p):
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)
    c = (-.007784894002430293, -.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (.007784695709041462, .3224671290700398, 2.445134137142996, 3.754408661907416)
    if p < .02425:
        q = math.sqrt(-2 * math.log(p)); s = 1
        return s * (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > .97575:
        q = math.sqrt(-2 * math.log(1-p)); s = -1
        return s * (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - .5; r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


@triton.jit
def _stats(x, cutoffs, n_cols: tl.constexpr, BLOCK: tl.constexpr, Z: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    v = tl.load(x + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(v, axis=0) / n_cols
    var = tl.sum(v * v, axis=0) / n_cols - mean * mean
    tl.store(cutoffs + row, mean + tl.sqrt(tl.maximum(var, 0.0)) * Z)


@triton.jit
def _apply(x, cutoffs, out, n: tl.constexpr, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x + offs, mask=mask).to(tl.float32)
    cutoff = tl.load(cutoffs + offs // n_cols, mask=mask)
    tl.store(out + offs, tl.maximum(v - cutoff, 0.0), mask=mask)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    if target_sparsity == 0.0:
        return inputs
    n_cols = inputs.shape[-1]
    rows = inputs.numel() // n_cols
    out = torch.empty_like(inputs)
    block = triton.next_power_of_2(n_cols)
    num_warps = 16 if rows >= 16384 else 8
    z = float(_ndtri_scalar(target_sparsity))
    _fused[(rows,)](inputs, out, n_cols=n_cols, BLOCK=block,
                    Z=z, num_warps=num_warps)
    return out
