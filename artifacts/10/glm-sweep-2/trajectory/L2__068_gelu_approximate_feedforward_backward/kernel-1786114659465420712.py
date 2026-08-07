import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_bwd_kernel(
    grad_gelu_out_ptr,        # [N, H]
    tanh_inner_ptr,           # [N, H]
    linear1_out_ptr,          # [N, H]
    grad_linear1_out_ptr,     # [N, H]
    N, H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    # Grid covers N*H elements, each program handles BLOCK elements
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N * H

    ggo = tl.load(grad_gelu_out_ptr + offs, mask=mask, other=0.0)
    ti = tl.load(tanh_inner_ptr + offs, mask=mask, other=0.0)
    l1o = tl.load(linear1_out_ptr + offs, mask=mask, other=0.0)

    # gelu_grad = 0.5*(1+tanh_inner) + 0.5*x*sech^2(inner)*d_inner/dx
    sech_sq = 1.0 - ti * ti
    d_inner_dx = 0.7978845608028654 * (1.0 + 0.134145 * l1o * l1o)
    gelu_grad = 0.5 * (1.0 + ti) + 0.5 * l1o * sech_sq * d_inner_dx

    result = ggo * gelu_grad
    tl.store(grad_linear1_out_ptr + offs, result, mask=mask)


def _gelu_bwd(grad_gelu_out, tanh_inner, linear1_out):
    grad_linear1_out = torch.empty_like(grad_gelu_out)
    N = grad_gelu_out.shape[0]
    H = grad_gelu_out.shape[1]
    n_el = N * H
    BLOCK = 2048
    grid = (triton.cdiv(n_el, BLOCK),)
    _gelu_bwd_kernel[grid](
        grad_gelu_out, tanh_inner, linear1_out, grad_linear1_out,
        n_el, H=H, BLOCK=BLOCK,
        num_stages=1, num_warps=8,
    )
    return grad_linear1_out


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
    # Backward through Linear2: output = gelu_out @ weight2.T
    grad_gelu_out = grad_output.matmul(weight2)
    grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
    gelu_out_2d = gelu_out.reshape(-1, gelu_out.shape[-1])
    grad_weight2 = grad_output_2d.t().matmul(gelu_out_2d)

    # Backward through GELU approximate (fused pointwise)
    grad_gelu_out_2d = grad_gelu_out.reshape(-1, grad_gelu_out.shape[-1])
    tanh_inner_2d = tanh_inner.reshape(-1, tanh_inner.shape[-1])
    linear1_out_2d = linear1_out.reshape(-1, linear1_out.shape[-1])
    grad_linear1_out_2d = _gelu_bwd(grad_gelu_out_2d, tanh_inner_2d, linear1_out_2d)
    grad_linear1_out = grad_linear1_out_2d.view(grad_gelu_out.shape)

    # Backward through Linear1: linear1_out = hidden_states @ weight1.T
    grad_hidden_states = grad_linear1_out.matmul(weight1)
    hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    grad_weight1 = grad_linear1_out_2d.t().matmul(hidden_states_2d)

    return grad_hidden_states, grad_weight1, grad_weight2
