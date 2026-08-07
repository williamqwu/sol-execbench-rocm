import torch

def _run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
    intermediate: torch.Tensor,
    intermediate_activated: torch.Tensor,
):
    batch_size, seq_len, hidden_size = grad_output.shape
    intermediate_size = intermediate.shape[-1]

    # Backward through second linear layer
    grad_intermediate_activated = torch.matmul(grad_output, weight2)  # [B, S, I]

    grad_output_reshaped = grad_output.reshape(-1, hidden_size)
    intermediate_activated_reshaped = intermediate_activated.reshape(-1, intermediate_size)
    grad_weight2 = torch.matmul(grad_output_reshaped.t(), intermediate_activated_reshaped)

    grad_bias2 = grad_output.sum(dim=[0, 1])

    # Backward through GELU activation (tanh approximation)
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715

    x = intermediate
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)
    tanh_inner = torch.tanh(inner)

    d_inner = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x)
    sech_squared = 1.0 - tanh_inner * tanh_inner

    gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_squared * d_inner

    grad_intermediate = grad_intermediate_activated * gelu_grad

    # Backward through first linear layer
    grad_hidden_states = torch.matmul(grad_intermediate, weight1)

    grad_intermediate_reshaped = grad_intermediate.reshape(-1, intermediate_size)
    hidden_states_reshaped = hidden_states.reshape(-1, hidden_size)
    grad_weight1 = torch.matmul(grad_intermediate_reshaped.t(), hidden_states_reshaped)

    grad_bias1 = grad_intermediate.sum(dim=[0, 1])

    return grad_hidden_states, grad_weight1, grad_bias1, grad_weight2, grad_bias2


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
    intermediate: torch.Tensor,
    intermediate_activated: torch.Tensor,
):
    return _compiled(grad_output, hidden_states, weight1, bias1, weight2, bias2, intermediate, intermediate_activated)


_compiled = torch.compile(_run, mode="default", fullgraph=True)
