import torch

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
    batch_size, seq_len, hidden_size = grad_output.shape
    intermediate_size = intermediate.shape[-1]

    side = torch.cuda.Stream()

    # grad_weight2 and grad_bias2 depend only on grad_output and intermediate_activated:
    # fully independent of the critical path (go@w2 -> gelu -> gi@w1).
    with torch.cuda.stream(side):
        grad_output_reshaped = grad_output.reshape(-1, hidden_size)
        intermediate_activated_reshaped = intermediate_activated.reshape(-1, intermediate_size)
        grad_weight2 = torch.matmul(grad_output_reshaped.t(), intermediate_activated_reshaped)
        grad_bias2 = grad_output.sum(dim=[0, 1])

    # Critical path
    grad_intermediate_activated = torch.matmul(grad_output, weight2)  # [B, S, I]

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

    grad_hidden_states = torch.matmul(grad_intermediate, weight1)

    # grad_weight1, grad_bias1 depend on grad_intermediate (critical path)
    grad_intermediate_reshaped = grad_intermediate.reshape(-1, intermediate_size)
    hidden_states_reshaped = hidden_states.reshape(-1, hidden_size)
    grad_weight1 = torch.matmul(grad_intermediate_reshaped.t(), hidden_states_reshaped)
    grad_bias1 = grad_intermediate.sum(dim=[0, 1])

    torch.cuda.current_stream().wait_stream(side)

    return grad_hidden_states, grad_weight1, grad_bias1, grad_weight2, grad_bias2
