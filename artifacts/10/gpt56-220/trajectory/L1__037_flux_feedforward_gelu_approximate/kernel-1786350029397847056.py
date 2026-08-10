import torch
import torch.nn.functional as F
import math

torch.backends.cuda.preferred_blas_library("cublas")

@torch.inference_mode()
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
    # Step 1: First linear projection [batch, seq, 3072] -> [batch, seq, 12288]
    x = F.linear(hidden_states, fc1_weight, fc1_bias)
    
    # Step 2: GELU activation with tanh approximation
    # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    x = torch.ops.aten.gelu_.default(x, approximate="tanh")
    
    # Step 3: Second linear projection [batch, seq, 12288] -> [batch, seq, 3072]
    output_shape = (*hidden_states.shape[:-1], fc2_weight.shape[0])
    output = torch.addmm(fc2_bias, x.reshape(-1, x.shape[-1]), fc2_weight.t())
    
    return output.reshape(output_shape)
