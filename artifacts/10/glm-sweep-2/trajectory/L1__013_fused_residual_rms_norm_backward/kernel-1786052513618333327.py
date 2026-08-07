import torch
import triton
import triton.language as tl


@triton.jit
def _grad_x_kernel(
    grad_output_ptr, normalized_ptr, rstd_ptr, weight_ptr,
    grad_hidden_states_ptr, grad_residual_ptr,
    n_rows, n_hidden,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    mask = offs_h < n_hidden

    go = tl.load(grad_output_ptr + row * n_hidden + offs_h, mask=mask, other=0.0).to(tl.float32)
    norm = tl.load(normalized_ptr + row * n_hidden + offs_h, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offs_h, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    gn = go * w
    dot = tl.sum(gn * norm, axis=0) / n_hidden
    gx = rstd * (gn - dot * norm)
    gxbf16 = gx.to(tl.bfloat16)

    out_off = row * n_hidden + offs_h
    tl.store(grad_hidden_states_ptr + out_off, gxbf16, mask=mask)
    tl.store(grad_residual_ptr + out_off, gxbf16, mask=mask)


@triton.jit
def _grad_weight_kernel(
    grad_output_ptr, normalized_ptr, grad_weight_ptr,
    n_rows, n_hidden,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_R
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < n_hidden

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for r in range(BLOCK_R):
        row = row_start + r
        if row < n_rows:
            go = tl.load(grad_output_ptr + row * n_hidden + offs_h, mask=mask_h, other=0.0).to(tl.float32)
            norm = tl.load(normalized_ptr + row * n_hidden + offs_h, mask=mask_h, other=0.0)
            acc += go * norm

    tl.atomic_add(grad_weight_ptr + offs_h, acc, mask=mask_h)


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor, normalized: torch.Tensor, rstd: torch.Tensor, weight: torch.Tensor):
    B, S, H = grad_output.shape
    n_rows = B * S
    n_hidden = H

    grad_weight = torch.zeros(n_hidden, dtype=torch.float32, device=grad_output.device)
    grad_hidden_states = torch.empty((B, S, H), dtype=torch.bfloat16, device=grad_output.device)
    grad_residual = torch.empty((B, S, H), dtype=torch.bfloat16, device=grad_output.device)

    BLOCK_H = _next_pow2(n_hidden)

    _grad_x_kernel[(n_rows,)](
        grad_output, normalized, rstd, weight,
        grad_hidden_states, grad_residual,
        n_rows, n_hidden,
        BLOCK_H=BLOCK_H,
        num_warps=8,
    )

    BLOCK_R = 16
    n_progs = (n_rows + BLOCK_R - 1) // BLOCK_R
    _grad_weight_kernel[(n_progs,)](
        grad_output, normalized, grad_weight,
        n_rows, n_hidden,
        BLOCK_H=BLOCK_H,
        BLOCK_R=BLOCK_R,
        num_warps=8,
    )
    return grad_hidden_states, grad_residual, grad_weight
