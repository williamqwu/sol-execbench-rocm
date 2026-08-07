import torch


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

    # Backward through GELU approximate activation
    gelu_grad_part1 = 0.5 * (1.0 + tanh_inner)
    sech_squared = 1.0 - tanh_inner * tanh_inner
    d_inner_dx = 0.7978845608028654 * (1.0 + 0.134145 * linear1_out * linear1_out)
    gelu_grad_part2 = 0.5 * linear1_out * sech_squared * d_inner_dx
    gelu_grad = gelu_grad_part1 + gelu_grad_part2
    grad_linear1_out = grad_gelu_out * gelu_grad

    # Backward through Linear1: linear1_out = hidden_states @ weight1.T
    grad_hidden_states = grad_linear1_out.matmul(weight1)
    grad_linear1_out_2d = grad_linear1_out.reshape(-1, grad_linear1_out.shape[-1])
    hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    grad_weight1 = grad_linear1_out_2d.t().matmul(hidden_states_2d)

    return grad_hidden_states, grad_weight1, grad_weight2
