import torch

@torch.no_grad()
def run(
    attention_weights: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """
    Compute fused attention output: attention_weights @ value with transpose and reshape.
    
    Args:
        attention_weights: Softmax-normalized attention scores [B, H, Q, K]
        value: Value matrix [B, H, K, D]
        
    Returns:
        output: Attention output [B, Q, H*D] ready for output projection
    """
    batch_size, num_heads, seq_len_q, seq_len_kv = attention_weights.shape
    attn_output = torch.bmm(
        attention_weights.reshape(batch_size * num_heads, seq_len_q, seq_len_kv),
        value.reshape(batch_size * num_heads, seq_len_kv, 64),
    )
    return attn_output.reshape(batch_size, num_heads, seq_len_q, 64).transpose(1, 2).contiguous().view(batch_size, seq_len_q, 1280)
