import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_rest_kernel(
    grad_gelu_ptr, x_ptr, tanh_out_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_gelu = tl.load(grad_gelu_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    tanh_out = tl.load(tanh_out_ptr + offsets, mask=mask)

    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715

    x_sq = x * x
    dtanh_arg_dx = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x_sq)
    sech_sq = 1.0 - tanh_out * tanh_out
    gelu_grad = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech_sq * dtanh_arg_dx

    out = grad_gelu * gelu_grad
    tl.store(out_ptr + offsets, out, mask=mask)


def _gelu_backward(grad_gelu, x):
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed)
    tanh_out = torch.tanh(tanh_arg)

    out = torch.empty_like(grad_gelu)
    n = grad_gelu.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    _gelu_rest_kernel[grid](grad_gelu, x, tanh_out, out, n, BLOCK_SIZE=1024)
    return out


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_output: torch.Tensor,
    gelu_output: torch.Tensor,
    fc2_weight: torch.Tensor,
    residual_output: torch.Tensor,
    normalized: torch.Tensor,
    var: torch.Tensor,
    ln_weight: torch.Tensor,
    eps: float,
):
    B, S, H = grad_output.shape
    I = fc1_weight.shape[0]

    # BACKWARD THROUGH LAYER NORM
    grad_ln_weight = (grad_output * normalized).sum(dim=(0, 1))
    grad_ln_bias = grad_output.sum(dim=(0, 1))
    grad_normalized = grad_output * ln_weight

    std = torch.sqrt(var + eps)
    grad_normalized_mean = grad_normalized.mean(dim=-1, keepdim=True)
    grad_normalized_normalized_mean = (grad_normalized * normalized).mean(dim=-1, keepdim=True)
    grad_residual_output = (1.0 / std) * (
        grad_normalized - grad_normalized_mean - normalized * grad_normalized_normalized_mean
    )

    # BACKWARD THROUGH RESIDUAL ADDITION
    grad_fc2_output = grad_residual_output
    grad_residual = grad_residual_output

    # BACKWARD THROUGH FC2
    grad_fc2_bias = grad_fc2_output.sum(dim=(0, 1))
    grad_fc2_output_reshaped = grad_fc2_output.view(-1, H)
    gelu_output_reshaped = gelu_output.view(-1, I)
    grad_fc2_weight = grad_fc2_output_reshaped.t() @ gelu_output_reshaped
    grad_gelu_output = grad_fc2_output @ fc2_weight

    # BACKWARD THROUGH GELU (fused)
    grad_fc1_output = _gelu_backward(grad_gelu_output, fc1_output)

    # BACKWARD THROUGH FC1
    grad_fc1_bias = grad_fc1_output.sum(dim=(0, 1))
    grad_fc1_output_reshaped = grad_fc1_output.view(-1, I)
    hidden_states_reshaped = hidden_states.view(-1, H)
    grad_fc1_weight = grad_fc1_output_reshaped.t() @ hidden_states_reshaped
    grad_hidden_states_fc1 = grad_fc1_output @ fc1_weight

    # COMBINE
    grad_hidden_states = grad_hidden_states_fc1 + grad_residual

    return (
        grad_hidden_states,
        grad_fc1_weight,
        grad_fc1_bias,
        grad_fc2_weight,
        grad_fc2_bias,
        grad_ln_weight,
        grad_ln_bias,
    )
