import torch

@torch.no_grad()
def run(
    attn_weights: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """
    Fused attention weights @ value matmul with transpose and reshape.
    
    Args:
        attn_weights: Attention weights after softmax, shape (batch, 40, seq_len, seq_len)
        value_states: Value states after GQA expansion, shape (batch, 40, seq_len, 128)
        
    Returns:
        Reshaped attention output ready for output projection, shape (batch, seq_len, 5120)
    """
    # Constants
    num_attention_heads = 40
    head_dim = 128
    hidden_size = 5120
    
    # Step 1: Attention weights @ value states
    # (batch, 40, seq_len, seq_len) @ (batch, 40, seq_len, 128) -> (batch, 40, seq_len, 128)
    batch_size, _, seq_len, _ = attn_weights.shape
    attn_output = torch.bmm(
        attn_weights.view(batch_size * 40, seq_len, seq_len),
        value_states.view(batch_size * 40, seq_len, 128),
    ).view(batch_size, 40, seq_len, 128)
    
    # Step 2: Transpose to move heads dimension
    # (batch, 40, seq_len, 128) -> (batch, seq_len, 40, 128)
    attn_output = attn_output.transpose(1, 2)
    
    # Step 3: Reshape to combine heads
    # (batch, seq_len, 40, 128) -> (batch, seq_len, 5120)
    attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
    
    # Ensure contiguous for next operation
    attn_output = attn_output.contiguous()
    
    return attn_output
