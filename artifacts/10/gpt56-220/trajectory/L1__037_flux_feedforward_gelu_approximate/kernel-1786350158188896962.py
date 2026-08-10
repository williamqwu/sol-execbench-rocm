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
    x = torch.addmm(
        fc1_bias,
        hidden_states.reshape(-1, hidden_states.shape[-1]),
        fc1_weight.t(),
    ).reshape(*hidden_states.shape[:-1], fc1_weight.shape[0])
    
    # Step 2: GELU activation with tanh approximation
    # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    x = torch.ops.aten.gelu_.default(x, approximate="tanh")
    
    # Step 3: Second linear projection [batch, seq, 12288] -> [batch, seq, 3072]
    output = F.linear(x, fc2_weight, fc2_bias)
    
    return output
