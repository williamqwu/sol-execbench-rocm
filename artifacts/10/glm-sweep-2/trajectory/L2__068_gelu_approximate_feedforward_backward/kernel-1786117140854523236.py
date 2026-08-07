import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_bwd_kernel(
    grad_gelu_out_ptr,
    tanh_inner_ptr,
    linear1_out_ptr,
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
    # Bitwise-match the PyTorch eager reference: enable_fp_fusion=False disables
    # FMA contraction so each mul/add rounds separately, and the left-associative
    # evaluation order mirrors Python's operator precedence.
    p1 = 0.5 * (1.0 + ti)
    t2 = ti * ti
    sech = 1.0 - t2
    kx = 0.134145 * l1o
    kxx = kx * l1o
    inner = 1.0 + kxx
    didx = 0.7978845608028654 * inner
    p2a = 0.5 * l1o
    p2b = p2a * sech
    p2 = p2b * didx
    gg = p1 + p2
    tl.store(out_ptr + offs, ggo * gg, mask=mask)


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

    grad_linear1_out = torch.empty_like(grad_gelu_out)
    n_el = grad_gelu_out.numel()
    BLOCK = 2048
    _gelu_bwd_kernel[(triton.cdiv(n_el, BLOCK),)](
        grad_gelu_out, tanh_inner, linear1_out, grad_linear1_out,
        n_el, BLOCK=BLOCK, num_warps=8, enable_fp_fusion=False,
    )

    grad_hidden_states = grad_linear1_out.matmul(weight1)
    hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    grad_linear1_out_2d = grad_linear1_out.reshape(-1, grad_linear1_out.shape[-1])
    grad_weight1 = grad_linear1_out_2d.t().matmul(hidden_states_2d)

    return grad_hidden_states, grad_weight1, grad_weight2
