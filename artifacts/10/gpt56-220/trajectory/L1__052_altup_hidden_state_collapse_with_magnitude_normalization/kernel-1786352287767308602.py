import torch

@torch.compile(fullgraph=True, mode="reduce-overhead")
@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    unembed_proj_1: torch.Tensor,
    unembed_proj_2: torch.Tensor,
    epsilon: float,
):
    """
    Collapses multiple AltUp prediction streams into a single hidden state.
    
    Args:
        hidden_states: [altup_num_inputs, batch_size, seq_len, hidden_size]
        unembed_proj_1: [hidden_size, hidden_size] - projection for stream 1
        unembed_proj_2: [hidden_size, hidden_size] - projection for stream 2
        epsilon: small value for numerical stability
    
    Returns:
        output: [batch_size, seq_len, hidden_size]
    """
    # Extract first stream as reference (no projection needed)
    first_stream = hidden_states[0]  # [batch, seq_len, hidden_size]
    
    # Compute target magnitude from first stream
    # Shape: [batch, seq_len, 1]
    target_magnitude = torch.sqrt(
        torch.mean(first_stream.to(torch.float32) ** 2, dim=-1, keepdim=True)
    )
    
    # Process stream 1
    # Project through unembed matrix: [batch, seq_len, hidden_size] @ [hidden_size, hidden_size].T
    projected_stream_1_bf = torch.matmul(hidden_states[1], unembed_proj_1.t())
    projected_stream_1 = projected_stream_1_bf.float()
    
    # Compute current magnitude
    inv_magnitude_1 = torch.rsqrt(torch.clamp_min(
        torch.mean(projected_stream_1 ** 2, dim=-1, keepdim=True), epsilon
    ))
    scale_1 = (target_magnitude * inv_magnitude_1).to(torch.bfloat16)
    normalized_stream_1 = projected_stream_1_bf * scale_1
    
    # Process stream 2
    projected_stream_2_bf = torch.matmul(hidden_states[2], unembed_proj_2.t())
    projected_stream_2 = projected_stream_2_bf.float()
    
    # Compute current magnitude
    inv_magnitude_2 = torch.rsqrt(torch.clamp_min(
        torch.mean(projected_stream_2 ** 2, dim=-1, keepdim=True), epsilon
    ))
    scale_2 = (target_magnitude * inv_magnitude_2).to(torch.bfloat16)
    normalized_stream_2 = projected_stream_2_bf * scale_2
    output = (first_stream + normalized_stream_1 + normalized_stream_2) / 3.0
    
    return output.to(torch.bfloat16)
