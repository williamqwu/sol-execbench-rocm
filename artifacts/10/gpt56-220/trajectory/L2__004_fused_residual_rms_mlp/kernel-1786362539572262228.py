import torch
import torch.nn.functional as F


@torch.compile
def _swiglu(gate, up):
    return F.silu(gate) * up

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    eps: float,
):
    """
    Fused residual + RMSNorm + SwiGLU MLP.
    
    Args:
        hidden_states: Output from attention block [batch_size, seq_len, hidden_size]
        residual: Residual connection from before attention [batch_size, seq_len, hidden_size]
        norm_weight: RMSNorm weight [hidden_size]
        gate_proj_weight: Gate projection weight [intermediate_size, hidden_size]
        up_proj_weight: Up projection weight [intermediate_size, hidden_size]
        down_proj_weight: Down projection weight [hidden_size, intermediate_size]
        eps: Epsilon for numerical stability
        
    Returns:
        Output tensor after MLP [batch_size, seq_len, hidden_size]
    """
    torch.backends.cuda.preferred_blas_library("ck")

    # Step 1: Residual connection
    # Shape: [batch_size, seq_len, hidden_size]
    hidden_states = residual + hidden_states
    
    # Step 2: RMSNorm
    hidden_states = F.rms_norm(hidden_states, (hidden_states.shape[-1],), norm_weight, eps)
    
    
    # Step 3: SwiGLU MLP
    # Gate projection: [batch_size, seq_len, hidden_size] @ [hidden_size, intermediate_size]
    # Shape: [batch_size, seq_len, intermediate_size]
    gate_output = F.linear(hidden_states, gate_proj_weight)
    
    # Up projection
    # Shape: [batch_size, seq_len, intermediate_size]
    up_output = F.linear(hidden_states, up_proj_weight)
    
    # SwiGLU activation: SiLU(gate) * up
    # SiLU(x) = x * sigmoid(x)
    # Shape: [batch_size, seq_len, intermediate_size]
    intermediate = _swiglu(gate_output, up_output)
    
    # Down projection
    # Shape: [batch_size, seq_len, hidden_size]
    output = F.linear(intermediate, down_proj_weight)
    
    torch.backends.cuda.preferred_blas_library("cublaslt")
    return output
