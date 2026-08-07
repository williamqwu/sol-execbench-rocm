import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_bwd_rest_kernel(
    grad_gelu_out_ptr,
    tanh_inner_ptr,
    linear1_out_ptr,
    sech_ptr,
    didx_ptr,
    out_ptr,
    n_el,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_el
    ggo = tl.load(grad_gelu_out_ptr + offs, mask=mask, other=0.0)
    ti = tl.load(tanh_inner_ptr + offs, mask=mask, other=0.0)
    l1o = tl.load(linear1_out_ptr + offs, mask=mask, other=0.0)
    sech = tl.load(sech_ptr + offs, mask=mask, other=0.0)
    didx = tl.load(didx_ptr + offs, mask=mask, other=0.0)
    part1 = 0.5 * (1.0 + ti)
    part2 = 0.5 * l1o * sech * didx
    tl.store(out_ptr + offs, ggo * (part1 + part2), mask=mask)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    linear1_out: torch.Tensor,
    tanh_inner: torch.Tensor,
    gelu_out: torch.Tensor,
):
    grad_gelu_out = grad_output.matmul(weight2)
    grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
    gelu_out_2d = gelu_out.reshape(-1, gelu_out.shape[-1])
    grad_weight2 = grad_output_2d.t().matmul(gelu_out_2d)

    # sech_squared and d_inner_dx computed with PyTorch eager (mul+add, no FMA)
    # to stay bitwise-identical to the reference; the rest is fused in Triton.
    sech_squared = 1.0 - tanh_inner * tanh_inner
    d_inner_dx = 0.7978845608028654 * (1.0 + 0.134145 * linear1_out * linear1_out)

    grad_linear1_out = torch.empty_like(grad_gelu_out)
    n_el = grad_gelu_out.numel()
    BLOCK = 2048
    _gelu_bwd_rest_kernel[(triton.cdiv(n_el, BLOCK),)](
        grad_gelu_out, tanh_inner, linear1_out, sech_squared, d_inner_dx,
        grad_linear1_out, n_el, BLOCK=BLOCK, num_warps=8,
    )

    grad_hidden_states = grad_linear1_out.matmul(weight1)
    hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    grad_linear1_out_2d = grad_linear1_out.reshape(-1, grad_linear1_out.shape[-1])
    grad_weight1 = grad_linear1_out_2d.t().matmul(hidden_states_2d)

    return grad_hidden_states, grad_weight1, grad_weight2
