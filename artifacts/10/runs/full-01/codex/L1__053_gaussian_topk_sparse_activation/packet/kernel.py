import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gaussian_sparse_kernel(
    x_ptr,
    y_ptr,
    n_cols: tl.constexpr,
    z,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)

    inv_n = 1.0 / n_cols
    mean = tl.sum(x, axis=0) * inv_n
    mean_square = tl.sum(x * x, axis=0) * inv_n
    variance = tl.maximum(mean_square - mean * mean, 0.0)
    cutoff = mean + tl.sqrt(variance) * z
    y = tl.maximum(x - cutoff, 0.0)
    tl.store(y_ptr + row * n_cols + cols, y, mask=mask)


@triton.jit
def _gaussian_sparse_kernel_tiled(
    x_ptr,
    y_ptr,
    n_cols: tl.constexpr,
    z,
    CHUNKS: tl.constexpr,
    TILE: tl.constexpr,
):
    row_start = tl.program_id(0) * n_cols
    cols = tl.arange(0, TILE)

    x0 = tl.load(x_ptr + row_start + cols).to(tl.float32)
    total = x0
    total_square = x0 * x0
    if CHUNKS >= 2:
        x1 = tl.load(x_ptr + row_start + TILE + cols).to(tl.float32)
        total += x1
        total_square += x1 * x1
    if CHUNKS >= 3:
        x2 = tl.load(x_ptr + row_start + 2 * TILE + cols).to(tl.float32)
        total += x2
        total_square += x2 * x2
    if CHUNKS >= 4:
        x3 = tl.load(x_ptr + row_start + 3 * TILE + cols).to(tl.float32)
        total += x3
        total_square += x3 * x3

    inv_n = 1.0 / n_cols
    mean = tl.sum(total, axis=0) * inv_n
    mean_square = tl.sum(total_square, axis=0) * inv_n
    variance = tl.maximum(mean_square - mean * mean, 0.0)
    cutoff = mean + tl.sqrt(variance) * z

    tl.store(y_ptr + row_start + cols, tl.maximum(x0 - cutoff, 0.0))
    if CHUNKS >= 2:
        tl.store(y_ptr + row_start + TILE + cols, tl.maximum(x1 - cutoff, 0.0))
    if CHUNKS >= 3:
        tl.store(y_ptr + row_start + 2 * TILE + cols, tl.maximum(x2 - cutoff, 0.0))
    if CHUNKS >= 4:
        tl.store(y_ptr + row_start + 3 * TILE + cols, tl.maximum(x3 - cutoff, 0.0))


def _ndtri(p: float) -> float:
    a1 = -3.969683028665376e1
    a2 = 2.209460984245205e2
    a3 = -2.759285104469687e2
    a4 = 1.383577518672690e2
    a5 = -3.066479806614716e1
    a6 = 2.506628277459239
    b1 = -5.447609879822406e1
    b2 = 1.615858368580409e2
    b3 = -1.556989798598866e2
    b4 = 6.680131188771972e1
    b5 = -1.328068155288572e1
    c1 = -7.784894002430293e-3
    c2 = -3.223964580411365e-1
    c3 = -2.400758277161838
    c4 = -2.549732539343734
    c5 = 4.374664141464968
    c6 = 2.938163982698783
    d1 = 7.784695709041462e-3
    d2 = 3.224671290700398e-1
    d3 = 2.445134137142996
    d4 = 3.754408661907416

    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    if p <= 0.97575:
        q = p - 0.5
        r = q * q
        return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)*q / (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    if target_sparsity == 0.0:
        return inputs

    n_cols = inputs.shape[-1]
    n_rows = inputs.numel() // n_cols
    output = torch.empty_like(inputs)
    z = _ndtri(float(target_sparsity))
    if n_rows < 4096 and n_cols != 16384:
        _gaussian_sparse_kernel_tiled[(n_rows,)](
            inputs,
            output,
            n_cols=n_cols,
            z=z,
            CHUNKS=n_cols // 4096,
            TILE=4096,
            num_warps=4,
            num_stages=1,
        )
    else:
        num_warps = 16 if n_cols >= 12288 else 8
        _gaussian_sparse_kernel[(n_rows,)](
            inputs,
            output,
            n_cols=n_cols,
            z=z,
            BLOCK=triton.next_power_of_2(n_cols),
            num_warps=num_warps,
            num_stages=1,
        )
    return output
