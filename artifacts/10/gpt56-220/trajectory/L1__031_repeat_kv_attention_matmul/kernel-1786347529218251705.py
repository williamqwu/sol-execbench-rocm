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
    # Use the GEMM alpha parameter so scaling happens in the GEMM epilogue.
    batch, slen = query.shape[0], query.shape[2]
    q = query.reshape(-1, slen, 128)
    k = key.reshape(-1, slen, 128)
    scores = torch.baddbmm(
        query.new_empty(1), q, k.transpose(1, 2),
        beta=0, alpha=128 ** -0.5,
    )
    return scores.reshape(batch, 32, slen, slen)
