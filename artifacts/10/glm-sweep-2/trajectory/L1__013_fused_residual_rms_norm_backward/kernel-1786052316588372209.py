import torch
import triton
import triton.language as tl


@triton.jit
def _fused_backward_kernel(
    grad_output_ptr, normalized_ptr, rstd_ptr, weight_ptr,
    grad_hidden_states_ptr, grad_residual_ptr, grad_weight_ptr,
    n_rows, n_hidden,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    offs_h = tl.arange(0, BLOCK_H)
    mask = offs_h < n_hidden

    go = tl.load(grad_output_ptr + row * n_hidden + offs_h, mask=mask, other=0.0).to(tl.float32)
    norm = tl.load(normalized_ptr + row * n_hidden + offs_h, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offs_h, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    # grad_weight += grad_output * normalized
    gw = go * norm
    tl.atomic_add(grad_weight_ptr + offs_h, gw, mask=mask)

    # grad_normalized = grad_output * weight
    gn = go * w

    # mean(grad_normalized * normalized) over hidden
    dot = tl.sum(gn * norm, axis=0) / n_hidden

    # grad_x = rstd * (grad_normalized - dot * normalized)
    gx = rstd * (gn - dot * norm)
    gxbf16 = gx.to(tl.bfloat16)

    out_ptr = grad_hidden_states_ptr + row * n_hidden + offs_h
    tl.store(out_ptr, gxbf16, mask=mask)
    tl.store(grad_residual_ptr + row * n_hidden + offs_h, gxbf16, mask=mask)


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
    num_warps = 8 if BLOCK_H >= 1024 else 4

    _fused_backward_kernel[(n_rows,)](
        grad_output, normalized, rstd, weight,
        grad_hidden_states, grad_residual, grad_weight,
        n_rows, n_hidden,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )
    return grad_hidden_states, grad_residual, grad_weight
