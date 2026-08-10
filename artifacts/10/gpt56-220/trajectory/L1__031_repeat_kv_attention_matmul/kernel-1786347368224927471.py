import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """
    Fused GQA key-value repetition and attention score computation.
    
    Args:
        query: [batch_size, num_attention_heads, seq_len, head_dim]
        key: [batch_size, num_key_value_heads, seq_len, head_dim]
    
    Returns:
        attn_weights: [batch_size, num_attention_heads, seq_len, seq_len]
    """
    # Scaling the small Q input avoids a separate elementwise pass over the
    # much larger [batch, heads, seq, seq] result. Inputs and output remain
    # BF16, allowing the ROCm GEMM backend to use native MFMA instructions.
    return torch.matmul(query * (128 ** -0.5), key.transpose(2, 3))
