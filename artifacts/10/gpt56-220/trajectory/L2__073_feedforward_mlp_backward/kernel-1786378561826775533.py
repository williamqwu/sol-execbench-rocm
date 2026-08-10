import torch
import torch.nn.functional as F

_aux_stream = torch.cuda.Stream()

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
    """
    Backward pass for feedforward MLP.
    
    Forward was:
        intermediate = hidden_states @ weight1.T + bias1
        intermediate_activated = gelu(intermediate)
        output = intermediate_activated @ weight2.T + bias2
    
    Args:
        grad_output: [B, S, H] gradient w.r.t. output
        hidden_states: [B, S, H] original input
        weight1: [I, H] first linear weight
        bias1: [I] first linear bias
        weight2: [H, I] second linear weight
        bias2: [H] second linear bias
        intermediate: [B, S, I] pre-activation values
        intermediate_activated: [B, S, I] post-activation values
    
    Returns:
        grad_hidden_states, grad_weight1, grad_bias1, grad_weight2, grad_bias2
    """
    batch_size, seq_len, hidden_size = grad_output.shape
    intermediate_size = intermediate.shape[-1]
    
    # Backward through second linear layer
    # output = intermediate_activated @ weight2.T + bias2
    # grad_intermediate_activated = grad_output @ weight2
    grad_output_reshaped = grad_output.reshape(-1, hidden_size)  # [B*S, H]
    intermediate_activated_reshaped = intermediate_activated.reshape(-1, intermediate_size)  # [B*S, I]
    current_stream = torch.cuda.current_stream()
    _aux_stream.wait_stream(current_stream)
    with torch.cuda.stream(_aux_stream):
        grad_weight2 = torch.matmul(grad_output_reshaped.t(), intermediate_activated_reshaped)
        grad_bias2 = grad_output.sum(dim=[0, 1])

    grad_intermediate_activated = torch.matmul(grad_output, weight2)  # [B, S, I]
    
    # Backward through GELU activation
    # GELU(x) = x * Phi(x) where Phi is standard normal CDF
    # d/dx GELU(x) = Phi(x) + x * phi(x) where phi is standard normal PDF
    # Using PyTorch's approximation
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coeff = 0.044715
    
    if batch_size * seq_len <= 1024:
        grad_intermediate = torch.ops.aten.gelu_backward(
            grad_intermediate_activated, intermediate, approximate="tanh")
    else:
        x = intermediate
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + coeff * x_cubed)
        tanh_inner = torch.tanh(inner)
        d_inner = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x)
        sech_squared = 1.0 - tanh_inner * tanh_inner
        gelu_grad = (0.5 * (1.0 + tanh_inner) +
                     0.5 * x * sech_squared * d_inner)
        grad_intermediate = grad_intermediate_activated * gelu_grad
    
    # Backward through first linear layer
    # intermediate = hidden_states @ weight1.T + bias1
    # grad_hidden_states = grad_intermediate @ weight1
    grad_hidden_states = torch.matmul(grad_intermediate, weight1)  # [B, S, H]
    
    # grad_weight1 = grad_intermediate.T @ hidden_states
    # Reshape: [B*S, I].T @ [B*S, H] -> [I, H]
    grad_intermediate_reshaped = grad_intermediate.reshape(-1, intermediate_size)  # [B*S, I]
    hidden_states_reshaped = hidden_states.reshape(-1, hidden_size)  # [B*S, H]
    grad_weight1 = torch.matmul(grad_intermediate_reshaped.t(), hidden_states_reshaped)  # [I, H]
    
    # grad_bias1 = sum(grad_intermediate) over batch and seq dims
    grad_bias1 = grad_intermediate.sum(dim=[0, 1])  # [I]
    
    current_stream.wait_stream(_aux_stream)
    return grad_hidden_states, grad_weight1, grad_bias1, grad_weight2, grad_bias2
