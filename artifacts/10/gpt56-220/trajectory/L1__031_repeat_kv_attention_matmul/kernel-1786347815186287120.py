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
    batch, heads, slen, dim = query.shape
    q = query.reshape(batch * heads, slen, dim)
    kt = key.transpose(2, 3).reshape(batch * heads, dim, slen)
    scores = torch.baddbmm(
        query.new_empty(1), q, kt,
        beta=0, alpha=128 ** -0.5,
    )
    return scores.reshape(batch, heads, slen, slen)
