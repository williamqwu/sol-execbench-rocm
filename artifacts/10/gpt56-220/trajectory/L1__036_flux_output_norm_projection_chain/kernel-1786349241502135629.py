import torch
import torch.nn.functional as F

_modulation_stream = torch.cuda.Stream()

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    proj_out_weight: torch.Tensor,
    proj_out_bias: torch.Tensor,
    eps: float,
):
    """
    Flux output normalization and projection chain.
    
    Args:
        hidden_states: Input tensor of shape (batch_size, seq_len, inner_dim)
        temb: Timestep embeddings of shape (batch_size, inner_dim)
        linear_weight: Weight for modulation projection (2*inner_dim, inner_dim)
        linear_bias: Bias for modulation projection (2*inner_dim,)
        proj_out_weight: Weight for output projection (output_dim, inner_dim)
        proj_out_bias: Bias for output projection (output_dim,)
        eps: Epsilon for LayerNorm numerical stability
        
    Returns:
        Output tensor of shape (batch_size, seq_len, output_dim)
    """
    token_rows = hidden_states.shape[0] * hidden_states.shape[1]
    use_overlap = token_rows >= 3072 or (token_rows >= 2048 and hidden_states.shape[0] <= 2)
    if use_overlap:
        current_stream = torch.cuda.current_stream()
        _modulation_stream.wait_stream(current_stream)
        with torch.cuda.stream(_modulation_stream):
            temb_silu = temb * torch.sigmoid(temb)
            modulation = F.linear(temb_silu, linear_weight, linear_bias)

    # Step 1: Layer normalization without affine parameters
    # Compute mean and variance along the last dimension
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
    var.add_(eps).sqrt_()
    hidden_states_norm = hidden_states - mean
    hidden_states_norm.div_(var)
    
    # Step 2: SiLU activation on temb followed by linear projection
    # SiLU: x * sigmoid(x)
    if use_overlap:
        current_stream.wait_stream(_modulation_stream)
    else:
        temb_silu = temb * torch.sigmoid(temb)
        modulation = F.linear(temb_silu, linear_weight, linear_bias)
    
    # Split into shift and scale
    inner_dim = temb.shape[-1]
    shift = modulation[:, :inner_dim]  # (batch_size, inner_dim)
    scale = modulation[:, inner_dim:]  # (batch_size, inner_dim)
    
    # Step 3: Apply adaptive modulation
    # Expand shift and scale to match hidden_states shape
    shift = shift.unsqueeze(1)  # (batch_size, 1, inner_dim)
    scale = scale.unsqueeze(1)  # (batch_size, 1, inner_dim)
    scale.add_(1.0)
    hidden_states_norm.mul_(scale)
    hidden_states_mod = hidden_states_norm.add_(shift)
    
    # Step 4: Large output projection
    # (batch_size, seq_len, inner_dim) @ (inner_dim, output_dim) + bias
    output = F.linear(hidden_states_mod, proj_out_weight, proj_out_bias)
    
    return output
