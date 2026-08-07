import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    grad_output_ptr, normalized_ptr, rstd_ptr, weight_ptr,
    grad_hidden_states_ptr, grad_residual_ptr, grad_weight_ptr,
    n_rows, n_hidden,
    BLOCK_H: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < n_hidden
    w = tl.load(weight_ptr + offs_h, mask=mask_h, other=0.0)

    gw_acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for r in range(ROWS_PER_PROG):
        row = row_start + r
        if row < n_rows:
            base = row * n_hidden + offs_h
            go = tl.load(grad_output_ptr + base, mask=mask_h, other=0.0).to(tl.float32)
            norm = tl.load(normalized_ptr + base, mask=mask_h, other=0.0)
            rstd = tl.load(rstd_ptr + row).to(tl.float32)

            gn = go * w
            gw_acc += go * norm

            dot = tl.sum(gn * norm, axis=0) / n_hidden
            gx = rstd * (gn - dot * norm)
            gxbf16 = gx.to(tl.bfloat16)

            tl.store(grad_hidden_states_ptr + base, gxbf16, mask=mask_h)
            tl.store(grad_residual_ptr + base, gxbf16, mask=mask_h)

    tl.atomic_add(grad_weight_ptr + offs_h, gw_acc, mask=mask_h)


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
    ROWS_PER_PROG = 16
    n_progs = (n_rows + ROWS_PER_PROG - 1) // ROWS_PER_PROG

    _fused_kernel[(n_progs,)](
        grad_output, normalized, rstd, weight,
        grad_hidden_states, grad_residual, grad_weight,
        n_rows, n_hidden,
        BLOCK_H=BLOCK_H,
        ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=8,
    )
    return grad_hidden_states, grad_residual, grad_weight
