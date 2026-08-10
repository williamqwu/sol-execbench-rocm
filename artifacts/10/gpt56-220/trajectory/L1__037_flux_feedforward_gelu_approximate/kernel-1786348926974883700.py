import torch
import torch.nn.functional as F
import math

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
):
    """
    FLUX FeedForward with GELU approximate activation.
    
    Architecture:
    - Linear: hidden_dim (3072) -> mlp_hidden_dim (12288)
    - GELU activation with tanh approximation
    - Linear: mlp_hidden_dim (12288) -> hidden_dim (3072)
    
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    shape = hidden_states.shape
    hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    x = torch.addmm(fc1_bias, hidden_states_2d, fc1_weight.t())
    
    # Step 2: GELU activation with tanh approximation
    # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    x = F.gelu(x, approximate="tanh")
    
    # Step 3: Second linear projection [batch, seq, 12288] -> [batch, seq, 3072]
    output = torch.addmm(fc2_bias, x, fc2_weight.t())
    
    return output.reshape(shape)
